"""Train SARGrayColor on MRSHaze; see train_kaggle.ipynb."""
import argparse
import copy
import itertools
import json
import math
import platform
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from graytocolor.color import rgb_to_ycbcr
from graytocolor.data import (MRSHazeDataset, dataset_options, make_split,
                             paired_records, write_manifest)
from graytocolor.losses import RestorationLoss
from graytocolor.metrics import image_metrics, psnr, ssim
from graytocolor.model import SARGrayColor
from graytocolor.runtime import (append_json, assert_resume_compatible, load_checkpoint,
                                load_config, restore_rng, rng_state, save_checkpoint,
                                seed_everything, seed_worker, select_device, validate_config)


def phase_plan(config):
    t = config['train']
    lengths = [(name, t[f'{name}_epochs']) for name in ('gray', 'color', 'joint')]
    if config['mode'] == 'e2e':
        lengths = [('joint', sum(n for _, n in lengths))]
    plan = []
    for name, count in lengths:
        plan.extend((name, epoch, count) for epoch in range(count))
    return plan


def set_phase(model, phase):
    for parameter in model.structure.parameters():
        parameter.requires_grad_(phase != 'color')
    for parameter in model.colorizer.parameters():
        parameter.requires_grad_(phase != 'gray')


def learning_rate(t, phase_epoch, phase_epochs, progress):
    position = phase_epoch + progress
    warmup = min(t['warmup_epochs'], max(0, phase_epochs - 1))
    if warmup and position < warmup:
        return t['min_lr'] + (t['lr']-t['min_lr']) * (position + 1e-3) / warmup
    fraction = min(1, max(0, (position-warmup) / max(1, phase_epochs-warmup)))
    return t['min_lr'] + 0.5 * (t['lr']-t['min_lr']) * (1 + math.cos(math.pi*fraction))


@torch.inference_mode()
def validate(model, loader, device, phase, deadline=None):
    model.eval()
    totals, count = {}, 0
    for batch in loader:
        if deadline is not None and time.monotonic() >= deadline:
            return None  # Partial validation must not select a best model.
        hazy, sar, gt = [batch[k].to(device) for k in ('hazy', 'sar', 'gt')]
        output = model(hazy, sar, structure_only=phase == 'gray')
        if phase == 'gray':
            gray_gt = rgb_to_ycbcr(gt)[:, :1]
            metrics = {'gray_psnr': psnr(output['gray'], gray_gt),
                       'gray_ssim': ssim(output['gray'], gray_gt)}
        else:
            metrics = image_metrics(output['rgb'], gt, output['gray'])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0) + value.sum().item()
        count += gt.shape[0]
    if deadline is not None and time.monotonic() >= deadline:
        return None
    return {key: value / count for key, value in totals.items()}


def train(config, resume=None, init_weights=None, stop_after_epoch=None):
    """stop_after_epoch saves a complete epoch; total schedule stays unchanged."""
    validate_config(config)
    session_started = time.monotonic()
    hours = config['train'].get('max_hours')
    deadline = session_started + hours * 3600 if hours is not None else None
    if resume and init_weights:
        raise ValueError('Use either --resume or --init-weights')
    seed_everything(config['seed'])
    device = select_device(config['device'])
    run_dir = Path(config['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume and (run_dir / 'last.pt').exists():
        raise FileExistsError(f'{run_dir}/last.pt exists; use --resume or a new run directory')
    checkpoint = load_checkpoint(resume) if resume else None
    if checkpoint:
        assert_resume_compatible(config, checkpoint['config'])
    records = paired_records(config['data']['root'], 'train')
    train_records, val_records, manifest = make_split(
        records, config['data']['val_fraction'], config['seed'],
        manifest=checkpoint['split'] if checkpoint else None)
    write_manifest(run_dir / 'split.json', manifest)
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    t = config['train']
    train_ds = MRSHazeDataset(train_records, crop_size=config['data']['crop_size'],
                             augment=True, **dataset_options(config))
    if t['max_val_images']:
        val_records = val_records[:t['max_val_images']]
    # Validation uses whole native-resolution images, never random crops.
    val_ds = MRSHazeDataset(val_records, **dataset_options(config))
    generator = torch.Generator().manual_seed(config['seed'])
    loader = DataLoader(train_ds, batch_size=t['batch_size'], shuffle=True,
                        num_workers=t['num_workers'], pin_memory=device.type == 'cuda',
                        worker_init_fn=seed_worker, generator=generator, drop_last=False)
    # batch_size=1 handles variable image sizes. A separate generator prevents
    # validation loader creation from affecting the training random stream.
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0,
                            generator=torch.Generator().manual_seed(config['seed']))
    model = SARGrayColor(**config['model']).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.structure.parameters(), 'name': 'structure'},
        {'params': model.colorizer.parameters(), 'name': 'colorizer'}],
        lr=t['lr'], weight_decay=t['weight_decay'])
    amp = t['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start_epoch, best = 0, {}
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scaler']:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, best = checkpoint['epoch'], checkpoint['best']
        restore_rng(checkpoint['rng'], generator)
    elif init_weights:
        initial = load_checkpoint(init_weights)
        model.load_state_dict(initial['model'])
    plan = phase_plan(config)
    if start_epoch >= len(plan):
        print(f'All {len(plan)} epochs already completed.', flush=True)
        return
    if stop_after_epoch is not None and stop_after_epoch <= start_epoch:
        raise ValueError('--stop-after-epoch must exceed completed epoch count')
    if checkpoint:
        # New Kaggle sessions mount previous outputs read-only. Carry forward
        # their history and best models, even if this session finds no new best.
        source = Path(resume).parent
        for name in ('best.pt', 'best_gray.pt', 'best_color.pt', 'best_joint.pt'):
            old_path, new_path = source / name, run_dir / name
            if old_path.is_file() and not new_path.exists():
                old_best = load_checkpoint(old_path)
                assert_resume_compatible(config, old_best['config'])
                if old_best['epoch'] <= start_epoch and old_best['split'] == manifest:
                    shutil.copy2(old_path, new_path)
        history = source / 'metrics.jsonl'
        destination = run_dir / 'metrics.jsonl'
        if history.is_file() and not destination.exists():
            for line in history.read_text().splitlines():
                record = json.loads(line)
                if record['epoch'] <= start_epoch:
                    append_json(destination, record)
    metadata = {'torch': str(torch.__version__), 'python': platform.python_version(),
                'device': str(device), 'parameters': sum(p.numel() for p in model.parameters()),
                'train_images': len(train_ds), 'val_images': len(val_ds)}
    (run_dir / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata), flush=True)
    print(f'Schedule: {config["mode"]}; {len(plan)} epochs; test split is not used.', flush=True)
    if hours is not None:
        print(f'Session training budget: {hours:g} hours, including validation.', flush=True)
    if t['max_train_batches'] or t['max_val_images']:
        print('DEVELOPMENT SUBSET ENABLED: results are not a full-dataset benchmark.', flush=True)
    batches = min(len(loader), t['max_train_batches'] or len(loader))
    completed_epoch = start_epoch
    last_epoch_seconds = checkpoint.get('metrics', {}).get('seconds') if checkpoint else None

    def checkpoint_state(epoch, phase, metrics):
        return {'format_version': 1, 'config': copy.deepcopy(config), 'epoch': epoch,
                'phase': phase, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(), 'best': best, 'split': manifest,
                'rng': rng_state(generator), 'metrics': metrics}

    # Copy a mounted resume checkpoint into the new session's output. An
    # epoch-zero checkpoint allows restarting if the first epoch exceeds budget.
    save_checkpoint(run_dir / 'last.pt', checkpoint_state(
        start_epoch, checkpoint['phase'] if checkpoint else plan[0][0],
        checkpoint.get('metrics', {}) if checkpoint else {}))

    def stop_for_time(reason):
        status = {'reason': reason, 'completed_epoch': completed_epoch,
                  'elapsed_hours': (time.monotonic()-session_started)/3600,
                  'resume_checkpoint': str(run_dir / 'last.pt')}
        (run_dir / 'session_stop.json').write_text(json.dumps(status, indent=2) + '\n')
        print(f'Time budget stop: {reason}. Preserved epoch {completed_epoch} in '
              f'{run_dir}/last.pt; resume it in another Kaggle session.', flush=True)

    for epoch in range(start_epoch, len(plan)):
        if deadline is not None:
            remaining = deadline-time.monotonic()
            # Prefer stopping between epochs, allowing 25% timing variation and
            # one minute for writes. Batch/image checks cover slower epochs.
            estimate = 1.25*last_epoch_seconds + 60 if last_epoch_seconds else 0
            if remaining <= 0 or (estimate and remaining < estimate):
                stop_for_time('insufficient time for another complete epoch')
                return
        phase, phase_epoch, phase_epochs = plan[epoch]
        set_phase(model, phase)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        seen, updates = 0, 0
        started = time.monotonic()
        # One color warm-up epoch uses predictions only. Longer warm-ups end
        # with zero teacher mixing to match inference before joint fine-tuning.
        teacher_ratio = (t['teacher_start'] * (1-phase_epoch/(phase_epochs-1))
                         if phase == 'color' and phase_epochs > 1 else 0.0)
        for batch_index, batch in enumerate(itertools.islice(loader, batches)):
            if deadline is not None and time.monotonic() >= deadline:
                stop_for_time('budget reached during training; incomplete epoch discarded')
                return
            hazy, sar, gt = [batch[k].to(device, non_blocking=True)
                             for k in ('hazy', 'sar', 'gt')]
            available = torch.ones(hazy.shape[0], 1, 1, 1, device=device)
            if config['model']['use_sar'] and t['sar_dropout']:
                available = (torch.rand_like(available) >= t['sar_dropout']).to(sar.dtype)
                sar = sar * available
            lr = learning_rate(t, phase_epoch, phase_epochs, batch_index/batches)
            for group in optimizer.param_groups:
                group['lr'] = lr * (t['gray_joint_lr_scale']
                                     if phase == 'joint' and group['name'] == 'structure' else 1)
            detach = phase == 'color' or config['mode'] == 'modular'
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(hazy, sar, detach_structure=detach,
                               teacher_gray=rgb_to_ycbcr(gt)[:, :1] if teacher_ratio else None,
                               teacher_ratio=teacher_ratio, structure_only=phase == 'gray')
                loss, terms = RestorationLoss(config['loss'], config['model']['use_sar'])(
                    output, hazy, gt, phase, sar_available=available)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}, batch {batch_index+1}')
            # Scale the final incomplete accumulation group by its real size.
            group_start = (batch_index // t['accumulation_steps']) * t['accumulation_steps']
            group_size = min(t['accumulation_steps'], batches-group_start)
            scaler.scale(loss / group_size).backward()
            if (batch_index+1) % t['accumulation_steps'] == 0 or batch_index+1 == batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t['grad_clip'],
                                               error_if_nonfinite=not amp)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            bs = hazy.shape[0]
            seen += bs
            for key, value in {'total': loss.item(), **terms}.items():
                totals[key] = totals.get(key, 0) + value * bs
            if batch_index == 0 or (batch_index+1) % 100 == 0 or batch_index+1 == batches:
                print(f'Epoch {epoch+1}/{len(plan)} [{phase}] batch {batch_index+1}/{batches}'
                      f' loss={loss.item():.5f} lr={lr:.2e}', flush=True)
        boundary = epoch+1 == len(plan) or plan[epoch+1][0] != phase
        do_validation = (epoch+1) % t['val_every'] == 0 or boundary
        if deadline is not None and time.monotonic() >= deadline:
            stop_for_time('budget reached before validation; incomplete epoch discarded')
            return
        validation = validate(model, val_loader, device, phase, deadline) if do_validation else {}
        if validation is None:
            stop_for_time('budget reached during validation; incomplete epoch discarded')
            return
        metric_name = 'gray_psnr' if phase == 'gray' else 'psnr'
        improved = bool(validation) and validation[metric_name] > best.get(phase, -float('inf'))
        if improved:
            best[phase] = validation[metric_name]
        record = {'epoch': epoch+1, 'phase': phase, 'phase_epoch': phase_epoch+1,
                  'teacher_ratio': teacher_ratio, 'lr': lr, 'optimizer_updates': updates,
                  'seconds': time.monotonic()-started,
                  'train': {k: v/seen for k, v in totals.items()}, 'val': validation}
        append_json(run_dir / 'metrics.jsonl', record)
        print(json.dumps(record), flush=True)
        state = checkpoint_state(epoch+1, phase, record)
        save_checkpoint(run_dir / 'last.pt', state)
        if improved:
            save_checkpoint(run_dir / f'best_{phase}.pt', state)
            # Final model selection uses the final scheduled phase only.
            if phase == plan[-1][0]:
                save_checkpoint(run_dir / 'best.pt', state)
        completed_epoch = epoch+1
        last_epoch_seconds = time.monotonic()-started
        print(f'Saved {run_dir}/last.pt (completed epoch {completed_epoch}).', flush=True)
        if stop_after_epoch is not None and epoch+1 >= stop_after_epoch:
            print(f'Stopped after completed epoch {epoch+1}; resume from {run_dir}/last.pt', flush=True)
            break


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='mrshaze.yaml')
    parser.add_argument('--data-root')
    parser.add_argument('--run-dir')
    parser.add_argument('--device')
    parser.add_argument('--mode', choices=['hybrid', 'e2e', 'modular'])
    parser.add_argument('--resume', help='Resume full state and unchanged schedule')
    parser.add_argument('--init-weights', help='Start a new schedule from model weights only')
    parser.add_argument('--stop-after-epoch', type=int,
                        help='Absolute epoch count; preserves schedule for another Kaggle session')
    parser.add_argument('--max-hours', type=float, help='Per-session training/validation time budget')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    for key in ('run_dir', 'device', 'mode'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.data_root:
        config['data']['root'] = args.data_root
    if args.max_hours is not None:
        config['train']['max_hours'] = args.max_hours
    train(config, args.resume, args.init_weights, args.stop_after_epoch)
