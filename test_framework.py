"""Synthetic correctness tests; no MRSHaze images or GPU required."""
import copy
import importlib
import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from graytocolor.color import rgb_to_ycbcr, ycbcr_to_rgb
from graytocolor.data import (MRSHazeDataset, load_sar, make_split,
                             paired_records)
from graytocolor.inference import predict
from graytocolor.losses import RestorationLoss
from graytocolor.metrics import image_metrics
from graytocolor.model import SARGrayColor
from graytocolor.runtime import load_checkpoint, load_config
from train import phase_plan, train

torch.set_num_threads(1)


@pytest.fixture
def dataset_root(tmp_path):
    root = tmp_path / 'MRSHaze'
    rng = np.random.default_rng(5)
    for split, ids in [('train', range(1001, 1009)), ('test', range(1, 4))]:
        for directory in ('hazy', 'SAR', 'GT'):
            (root / split / directory).mkdir(parents=True)
        for key in ids:
            gt = rng.integers(20, 200, (17, 19, 3), dtype=np.uint8)
            hazy = np.rint(gt.astype(float)*0.65 + 80).astype(np.uint8)
            sar = gt.mean(-1).astype(np.uint8)
            for name, array in [('GT', gt), ('hazy', hazy), ('SAR', sar)]:
                Image.fromarray(array).save(root / split / name / f'{key:05d}.png')
    return root


def tiny_config(root, destination):
    config = load_config('mrshaze.yaml')
    config['data'].update(root=str(root), crop_size=16, val_fraction=0.25)
    config['model'].update(width=4, depth=1)
    config['train'].update(gray_epochs=1, color_epochs=2, joint_epochs=1,
                           batch_size=2, num_workers=0, amp=False,
                           accumulation_steps=2, sar_dropout=0.2)
    config.update(run_dir=str(destination), device='cpu', mode='hybrid')
    return config


def test_color_roundtrip_and_gamut():
    rgb = torch.rand(2, 3, 13, 17)
    ycc = rgb_to_ycbcr(rgb)
    torch.testing.assert_close(ycbcr_to_rgb(ycc[:, :1], ycc[:, 1:]), rgb, atol=2e-6, rtol=0)
    y = torch.rand(2, 1, 13, 17)
    chroma = torch.randn(2, 2, 13, 17) * 10
    result = ycbcr_to_rgb(y, chroma)
    assert result.min() >= -1e-6 and result.max() <= 1+1e-6
    torch.testing.assert_close(rgb_to_ycbcr(result)[:, :1], y, atol=2e-6, rtol=0)


@pytest.mark.parametrize('detach', [False, True])
def test_color_loss_gradient_routing(detach):
    torch.manual_seed(3)
    model = SARGrayColor(width=4, depth=1)
    hazy, sar = torch.rand(2, 3, 17, 19), torch.rand(2, 1, 17, 19)
    output = model(hazy, sar, detach_structure=detach)
    output['rgb'].square().mean().backward()
    gray_grads = [p.grad for p in model.structure.parameters() if p.grad is not None]
    if detach:
        assert not gray_grads
    else:
        assert sum(g.abs().sum().item() for g in gray_grads) > 1e-6
    assert model.colorizer.out.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.testing.assert_close(rgb_to_ycbcr(output['rgb'])[:, :1], output['gray'], atol=2e-6, rtol=0)


def test_teacher_is_training_only():
    model = SARGrayColor(width=4, depth=1).eval()
    with pytest.raises(ValueError, match='training-only'):
        model(torch.rand(1,3,16,16), torch.rand(1,1,16,16),
              teacher_gray=torch.rand(1,1,16,16), teacher_ratio=0.5)


def test_no_sar_ablation():
    model = SARGrayColor(width=4, depth=1, use_sar=False).eval()
    hazy = torch.rand(1,3,15,21)
    with torch.no_grad():
        a = model(hazy, torch.zeros(1,1,15,21))
        b = model(hazy, torch.rand(1,1,15,21))
    torch.testing.assert_close(a['rgb'], b['rgb'], rtol=0, atol=0)
    assert torch.count_nonzero(a['haze']) == 0


def test_strict_pairing_and_split(dataset_root):
    records = paired_records(dataset_root, 'train')
    a, b, manifest = make_split(records, 0.25, 42)
    a2, b2, _ = make_split(records, 0.25, 99, manifest=manifest)
    assert len(a) == 6 and len(b) == 2 and a == a2 and b == b2
    assert not set(manifest['train_ids']) & set(manifest['val_ids'])
    (dataset_root / 'train' / 'SAR' / '01001.png').unlink()
    with pytest.raises(ValueError, match='Unpaired'):
        paired_records(dataset_root, 'train')


def test_paired_geometry_and_small_images(dataset_root):
    records = paired_records(dataset_root, 'train')
    # Set all modalities to the same grayscale pattern to test synchronized
    # rotations, flips, padding and crops without relying on RNG choices.
    array = np.arange(12*14, dtype=np.uint8).reshape(12,14)
    for key in ('hazy','GT','SAR'):
        Image.fromarray(array).save(records[0][key])
    dataset = MRSHazeDataset(records[:1], crop_size=24, augment=True)
    for _ in range(8):
        item = dataset[0]
        assert item['hazy'].shape == (3,24,24)
        torch.testing.assert_close(item['hazy'][:1], item['sar'])
        torch.testing.assert_close(item['gt'], item['hazy'])


def test_sar_16bit_and_channel_validation(tmp_path):
    path = tmp_path / 'sar.png'
    Image.fromarray(np.array([[0,65535],[32768,1]], dtype=np.uint16)).save(path)
    sar = load_sar(path)
    assert sar[0,0,0] == 0 and sar[0,0,1] == 1
    torch.testing.assert_close(sar[0,1,0], torch.tensor(32768/65535))
    with pytest.raises(ValueError, match='channels'):
        load_sar(path, channels=2)


def test_metrics_identity_and_black():
    rgb = torch.rand(2,3,13,17)
    metrics = image_metrics(rgb,rgb,rgb_to_ycbcr(rgb)[:,:1])
    torch.testing.assert_close(metrics['ssim'], torch.ones(2))
    assert (metrics['psnr'] == 120).all()
    assert metrics['sam_deg'].max() < 1e-4
    for value in image_metrics(torch.zeros_like(rgb),torch.zeros_like(rgb)).values():
        assert torch.isfinite(value).all()


def test_all_losses_are_finite():
    model = SARGrayColor(width=4, depth=1)
    hazy, sar, target = torch.rand(2,3,16,16), torch.rand(2,1,16,16), torch.rand(2,3,16,16)
    loss, terms = RestorationLoss(load_config('mrshaze.yaml')['loss'])(
        model(hazy,sar),hazy,target,sar_available=torch.zeros(2,1,1,1))
    assert torch.isfinite(loss) and terms['gate'] == 0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_autocast_and_sar_gradient():
    model = SARGrayColor(width=4, depth=1)
    hazy, sar = torch.rand(2,3,17,19), torch.rand(2,1,17,19)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output = model(hazy, sar)
        loss, _ = RestorationLoss(load_config('mrshaze.yaml')['loss'])(output,hazy,torch.rand_like(hazy))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.structure.sar_stem.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


class PixelModel(torch.nn.Module):
    def forward(self, hazy, sar):
        return {'rgb':hazy, 'gray':hazy[:,:1], 'haze':sar[:,:1]}


def test_tiled_coverage():
    hazy, sar = torch.rand(1,3,19,39), torch.rand(1,1,19,39)
    result = predict(PixelModel(),hazy,sar,tile_size=16,overlap=5)
    torch.testing.assert_close(result['rgb'],hazy)
    torch.testing.assert_close(result['haze'],sar)
    with pytest.raises(ValueError, match='overlap'):
        predict(PixelModel(),hazy,sar,tile_size=16,overlap=16)


def test_phase_plans():
    config = load_config('mrshaze.yaml')
    assert config['mode'] == 'e2e'
    assert len(phase_plan(config)) == 6
    assert set(p[0] for p in phase_plan(config)) == {'joint'}
    config['train'].update(gray_epochs=10, color_epochs=10, joint_epochs=80)
    config['mode']='hybrid'
    assert len(phase_plan(config)) == 100
    config['mode']='e2e'
    assert set(p[0] for p in phase_plan(config)) == {'joint'}
    config['mode']='modular'
    assert phase_plan(config)[10][0] == 'color'


@pytest.mark.parametrize('phase', ['train', 'validation', 'between_epochs'])
def test_time_budget_preserves_resumable_checkpoint(dataset_root, tmp_path, monkeypatch, phase):
    module = importlib.import_module('train')
    config = tiny_config(dataset_root, tmp_path/'reference')
    config['mode'] = 'e2e'
    config['train'].update(gray_epochs=0, color_epochs=0, joint_epochs=2, max_hours=None)
    train(config)
    reference = load_checkpoint(tmp_path/'reference'/'last.pt')
    config['run_dir'] = str(tmp_path/'limited')
    config['train']['max_hours'] = 1
    clock = {'now': 0.0, 'train_batches': 0}
    original_forward = SARGrayColor.forward

    def forward(model, *args, **kwargs):
        output = original_forward(model, *args, **kwargs)
        if model.training:
            clock['train_batches'] += 1
            if phase == 'train' and clock['train_batches'] == 4:
                clock['now'] = 4000  # First batch of epoch two exceeds budget.
            if phase == 'between_epochs' and clock['train_batches'] == 3:
                clock['now'] = 2000  # Epoch one fits; another epoch would not.
        elif phase == 'validation':
            clock['now'] = 4000  # First validation image exceeds budget.
        return output

    with monkeypatch.context() as patch:
        patch.setattr(module, 'time', SimpleNamespace(monotonic=lambda: clock['now']))
        patch.setattr(SARGrayColor, 'forward', forward)
        train(config)
    limited = load_checkpoint(tmp_path/'limited'/'last.pt')
    expected_epoch = 0 if phase == 'validation' else 1
    assert limited['epoch'] == expected_epoch
    status = json.loads((tmp_path/'limited'/'session_stop.json').read_text())
    assert status['completed_epoch'] == expected_epoch
    if phase == 'validation':
        assert not (tmp_path/'limited'/'best.pt').exists()
        assert not (tmp_path/'limited'/'metrics.jsonl').exists()
    # Changing only the session budget is allowed, and resuming the preserved
    # state must undo all effects of discarded batches/partial validation.
    config['train']['max_hours'] = None
    config['run_dir'] = str(tmp_path/'continued')
    train(config, resume=tmp_path/'limited'/'last.pt')
    continued = load_checkpoint(tmp_path/'continued'/'last.pt')
    assert continued['epoch'] == reference['epoch'] == 2
    for key in reference['model']:
        torch.testing.assert_close(reference['model'][key], continued['model'][key], atol=0, rtol=0)
    rows = (tmp_path/'continued'/'metrics.jsonl').read_text().splitlines()
    assert len(rows) == 2
    assert (tmp_path/'continued'/'best.pt').exists()


@pytest.mark.parametrize('hours', [0, -1, float('inf'), float('nan'), True])
def test_invalid_time_budget(hours):
    from graytocolor.runtime import validate_config
    config = load_config('mrshaze.yaml')
    config['train']['max_hours'] = hours
    with pytest.raises(ValueError, match='max_hours'):
        validate_config(config)


def test_training_resume_and_inference(dataset_root,tmp_path):
    config = tiny_config(dataset_root,tmp_path/'full')
    train(config)
    full = load_checkpoint(tmp_path/'full'/'last.pt')
    partial_config = copy.deepcopy(config)
    partial_config['run_dir']=str(tmp_path/'resumed')
    train(partial_config,stop_after_epoch=2)
    train(partial_config,resume=tmp_path/'resumed'/'last.pt')
    resumed = load_checkpoint(tmp_path/'resumed'/'last.pt')
    assert full['epoch']==resumed['epoch']==4
    assert full['best']==resumed['best']
    assert full['split']==resumed['split']
    for key in full['model']:
        torch.testing.assert_close(full['model'][key],resumed['model'][key],atol=0,rtol=0)
    gray = load_checkpoint(tmp_path/'full'/'best_gray.pt')
    color = load_checkpoint(tmp_path/'full'/'best_color.pt')
    for key in gray['model']:
        if key.startswith('structure.'):
            torch.testing.assert_close(gray['model'][key],color['model'][key],atol=0,rtol=0)
    assert any(not torch.equal(gray['model'][k],color['model'][k])
               for k in gray['model'] if k.startswith('colorizer.'))
    assert (tmp_path/'full'/'best.pt').exists()
    from infer import run
    args = Namespace(weights=str(tmp_path/'full'/'best.pt'),data_root=str(dataset_root),
                     split='test',output_dir=str(tmp_path/'predictions'),device='cpu',
                     num_workers=0,tile_size=0,overlap=4,save_aux=True,no_gt=False,amp=False,limit=None)
    run(args)
    assert len(list((tmp_path/'predictions'/'rgb').glob('*.png')))==3
    assert len(list((tmp_path/'predictions'/'gray').glob('*.png')))==3
    assert (tmp_path/'predictions'/'summary.json').exists()
    # Ground-truth-free inference must work even when the GT directory is absent.
    for path in (dataset_root/'test'/'GT').glob('*.png'):
        path.rename(path.with_suffix('.hidden'))
    args.no_gt=True
    args.tile_size=16
    args.overlap=4
    args.output_dir=str(tmp_path/'without_gt')
    run(args)
    assert len(list((tmp_path/'without_gt'/'rgb').glob('*.png')))==3
    for path in (dataset_root/'test'/'GT').glob('*.hidden'):
        path.rename(path.with_suffix('.png'))
    from evaluate import evaluate
    result = evaluate(tmp_path/'predictions'/'rgb',dataset_root/'test'/'GT',tmp_path/'evaluated','cpu')
    assert result['images']==3
