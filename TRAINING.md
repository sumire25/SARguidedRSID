# Training protocol and Kaggle operations

This document describes the implemented behavior in `train.py`,
`graytocolor/`, `mrshaze.yaml`, and `train_kaggle.ipynb`. The model overview and
relationship to the supplied papers are in [README.md](README.md).

## Current experiment

The default is **E2E training for six epochs**, with an **eight-hour wall-clock
budget per training invocation**. It replaces the initial 10/10/80 hybrid
configuration because the reported grayscale-only run was already taking
approximately 27 minutes per training epoch before validation. The E2E path
also runs the colorizer and its backward pass, so the grayscale timing cannot
be used as its measured epoch duration.

Six epochs are an initial experiment sized for the available runtime, not a
convergence claim. The guard may stop before epoch six. Inspect validation
curves to decide whether to start a longer experiment across multiple sessions.

| Setting | Current value | Meaning |
| --- | --- | --- |
| Mode | `e2e` | Both networks learn jointly from the first batch |
| Phase lengths | `0 / 0 / 6` | Grayscale / color / joint epochs |
| Session budget | `8.0` hours | Includes training-script setup and validation |
| Model | Width 32, depth 2 | Three resolutions with 32/64/128 channels |
| Parameters | 1,326,414 | Default one-channel SAR model |
| Training batch | 4 | One GPU; no automatic multi-GPU execution |
| Crop | 256 × 256 | Random paired training crop |
| Accumulation | 1 | One optimizer update per training batch |
| Optimizer | AdamW | Weight decay `1e-4`; default betas `(0.9, 0.999)` and epsilon `1e-8` |
| Base LR | `2e-4` | Colorizer LR at the top of the schedule |
| Minimum base LR | `1e-6` | Floor of the base schedule |
| Structure LR multiplier | `0.5` | Applies throughout joint training, including E2E |
| LR warm-up | 1 epoch | Followed by cosine decay |
| Gradient clipping | Norm 1.0 | After unscaling AMP gradients |
| Mixed precision | Enabled on CUDA | fp16 autocast with gradient scaling |
| SAR dropout | 0.1 | Per-image probability during training |
| Loader workers | 2 training / 0 validation | Validation batch size is 1 |
| Validation interval | Every epoch | Also forced at phase boundaries |
| Seed | 42 | Split, initialization, shuffling, and augmentation seeds |

At 5,400 training images and batch size 4, a full epoch has **1,350 batches**.
Six completed epochs therefore contain 8,100 batches and, with accumulation 1,
8,100 attempted optimizer updates. AMP may skip an update on nonfinite
gradients; the logged update count counts attempts. Training uses all samples;
`drop_last=False`. There is no automatic early stopping based on validation
plateaus. The time budget and optional epoch stop are operational limits.

## What the model learns

Stage 1 receives hazy RGB and registered SAR. Its target is the optical GT's
BT.601 luminance, not SAR intensity. Stage 2 uses predicted luminance, hazy RGB
and chrominance, structure features, and the fusion-region map to predict
chrominance corrections. Gamut mapping contracts chroma while preserving
predicted luminance. There is no unrestricted final RGB refinement head.

E2E means final reconstruction errors can update both stages. Grayscale losses
are retained to keep the intermediate image meaningful. No GT image, GT
grayscale, or target-derived region label is provided to the model during
validation/testing. GT is used there only to calculate metrics.

The model starts from random weights unless `INIT_WEIGHTS` is set. There is no
pretrained backbone, GAN discriminator, diffusion sampler, perceptual network,
or external caption model. The default has one deterministic inference pass.

## Modes and phase behavior

| Mode | Stage warm-ups | Final phase | Can RGB losses update the structure network? |
| --- | --- | --- | --- |
| `e2e` | None; all configured epochs become joint epochs | Both networks train with all active losses | Yes, from epoch one |
| `hybrid` | Grayscale only, then color with structure frozen | Both networks train jointly | Yes, during joint phase |
| `modular` | Same warm-up sequence as hybrid | Both networks train, but all structure-to-color connections are detached | No |

The E2E epoch count is the **sum** of `gray_epochs`, `color_epochs`, and
`joint_epochs`. The default explicitly sets these to 0/0/6 to avoid ambiguity.
Changing only `MODE` to hybrid/modular while keeping 0/0/6 does not invent
warm-up epochs. For a short comparison, explicitly choose, for example, 2/1/3.

In the grayscale phase, the colorizer is frozen and not executed. In the color
phase, the entire structure network is frozen. In modular's final phase, the
structure network continues to learn from its own losses; it is not permanently
frozen. The colorizer sees updated structure predictions but sends no gradient
back through luminance, features, or the map.

If a color warm-up is configured, `teacher_start=0.5` mixes 50% GT grayscale
with 50% predicted grayscale initially, decreasing linearly to zero by its last
epoch. A single color warm-up epoch uses predictions only. The teacher never
supplies structure features or maps. **Default E2E has no color warm-up, so
teacher mixing is inactive despite this optional config value.** Joint
training, validation, and inference always use predicted grayscale.

## Dataset pairing and augmentation

All three PNG folders must have exactly the same filename stems. Missing/extra
modalities fail explicitly. The loader verifies equal spatial dimensions and
does not resize, align, or register modalities. Use registered triplets.

Optical RGB is converted from 8-bit PNG to float `[0,1]`. Default SAR is a
one-channel rendering: unsigned integer PNG values are divided by the dtype
maximum, and RGB SAR renderings are averaged across their RGB channels. This
is not a calibration procedure for raw SAR. No per-image contrast stretching
or min/max normalization is applied. Log compression defaults to off. See the
README for `sar_channels`, `sar_scale`, and `sar_log1p` options.

For each training sample load:

1. If necessary, replicate the bottom/right edge to make the image at least
   256 × 256. Apply the same padding to every modality.
2. Select a uniformly random 256 × 256 crop location, shared by all modalities.
3. Select a random rotation from 0°, 90°, 180°, and 270°.
4. Apply a horizontal flip with probability 0.5, after rotation. These choices
   span eight geometric orientations, including the equivalent vertical flips.
5. In the training loop, replace SAR with zeros independently per sample with
   probability 0.1. Optical input and GT are left intact. Dropped samples are
   excluded from fusion-region BCE supervision.

This is online augmentation: files are not duplicated, dataset length remains
5,400, and crops/orientations are sampled again on later loads. There is no
color jitter, synthetic haze generation, random rescaling, or added speckle.
If an image is exactly crop-sized, the crop location has no variation, although
rotations/flips still vary. SAR dropout is a training regularizer, not a claim
that inference is equally accurate without SAR.

## Split and validation protocol

The supplied 6,000-image training split is shuffled once with seed 42. A fixed
10% holdout gives **5,400 optimization images and 600 validation images**.
`split.json` and every checkpoint record the IDs. Resume uses the saved IDs.
No test images are read by training, split creation, or model selection. The
2,000-image test partition is reserved for the separate testing notebook.

This split is random by image ID; it is not stratified by haze level and does
not enforce geographic separation. IDs alone cannot establish scene
independence. Use source-scene metadata if geographic leakage must be ruled out.

Validation uses **full native-resolution images**, batch size 1, eval mode, and
no gradient updates. It has no crops, padding augmentation, rotations, flips,
teacher mixing, or SAR dropout. It is fp32 by default even when training uses
AMP. Its separate loader generator does not alter training's random stream.
`max_val_images` and `max_train_batches` default to null; setting either creates
a development subset that must not be reported as the full benchmark.

Grayscale phases report luminance PSNR/SSIM. Color/joint phases report RGB
PSNR, SSIM, MAE, RGB SAM in degrees, ERGAS, and recovered-luminance PSNR/SSIM.
Metrics are arithmetic means of per-image scores, not a PSNR calculated from
one global dataset MSE. `best_gray.pt` uses grayscale PSNR; `best_color.pt` and
`best_joint.pt` use RGB PSNR. `best.pt` is selected only in the final configured
phase. With default E2E, it is available after the first completed validation.

SSIM uses an 11×11 Gaussian valid window, sigma 1.5, and data range 1. There is
no border shaving beyond the valid SSIM window, and no reference resizing. RGB
SAM is a three-channel color measure, not hyperspectral evaluation. ERGAS uses
spatial ratio 1 and a reference-channel mean floor of `1e-6`. Inference scores
use float RGB before PNG quantization; `evaluate.py` scores exported 8-bit PNGs,
so the two protocols can differ slightly. Report which one was used.

## Losses and optimizer details

All reconstruction losses are calculated in fp32 under AMP. Global channel
attention correlations and softmax are also calculated in fp32. The structure
and colorizer use channel normalization, not BatchNorm running statistics.

| Loss | Default weight | Definition / target |
| --- | ---: | --- |
| Grayscale | 1.0 | Mean absolute error against GT luminance |
| Grayscale SSIM | 0.2 | Mean `1 − SSIM` |
| Edge | 0.1 | Average L1 of horizontal and vertical luminance differences |
| RGB | 1.0 | Mean absolute RGB reconstruction error |
| Chroma | 0.5 | L1 on gamut-mapped output Cb/Cr against GT Cb/Cr |
| RGB SSIM | 0.2 | Mean `1 − SSIM` |
| Frequency | 0.05 | Mean magnitude of the complex `rfft2` difference, orthonormal FFT |
| Region gate | 0.05 | BCE across the three fusion-map resolutions, averaged |

The frequency loss compares complex coefficients, including phase information;
it is not a comparison of FFT magnitudes alone. The gate label is the detached
proxy `clip(mean(abs(hazy − GT), RGB) / 0.25, 0, 1)`, area-resized to each map.
It represents optical discrepancy, **not physical transmission or a measured
haze mask**. A zero-SAR ablation disables the gate loss. Default E2E uses all
these losses; color-only phases omit the grayscale/edge/gate terms, and
grayscale-only phases omit RGB/chroma/RGB-SSIM/frequency terms.

The base LR is evaluated from fractional epoch position on every batch. It
warms up from approximately `1e-6` toward `2e-4` over epoch one, then follows
cosine decay toward `1e-6` over the remaining schedule. Discrete batches do not
necessarily land exactly on the continuous schedule's endpoint. Joint
structure LR is multiplied by 0.5, so its peak is `1e-4`. The logged `lr` is the
base/colorizer LR. Each explicitly configured stage gets its own schedule.

If accumulation exceeds one, each microbatch loss is divided by the number of
microbatches in that update group, including the last short group. Updates occur
after accumulation, unscaling, and gradient clipping. Accumulation does not
combine both Kaggle GPUs and does not make arbitrary batch-size changes
equivalent experiments. Nonfinite losses raise an error; AMP can skip optimizer
steps with nonfinite gradients.

Batch log lines show that batch's loss, so fluctuations are expected.
`metrics.jsonl` records sample-weighted epoch mean losses, validation scores,
phase, LR, teacher ratio, elapsed epoch seconds, and optimizer update attempts.
Validation numbers are the useful checkpoint-selection signal. Duplicate lines
in a Kaggle viewer alone do not establish duplicate training processes.

## Time guard: exact behavior

Set `MAX_TRAIN_HOURS` in the notebook or `train.max_hours` in YAML. A CLI
override is available:

```bash
python train.py --config mrshaze.yaml --max-hours 8
```

Use YAML `null` or notebook `None` to disable it. A positive finite number of
hours is required when enabled. There is no extra monitoring process or GPU
computation: the guard reads a monotonic clock between batches and validation
images. This overhead is negligible relative to forward/backward computation.

The eight-hour budget starts on entry to the training function. It includes
model/dataset setup, training, validation, and elapsed checkpoint handling. It
does not include notebook cloning/install before the training script or plots
and Kaggle artifact publication afterward. The budget is intentionally shorter
than the normal notebook session allowance to leave room for those operations.
Check your account's current runtime/quota rather than treating eight hours as
a universal Kaggle guarantee.

Before a subsequent epoch starts, remaining time is compared with **1.25 × the
previous epoch duration + 60 seconds**. If it likely will not fit, training
stops between epochs. The estimate is a heuristic; stage transitions and I/O
variation can change epoch cost. The first epoch has no measured estimate.

If time expires during training or validation, training returns normally at the
next check. The unfinished epoch is discarded. `last.pt` retains the previous
completed epoch and its optimizer/RNG states. **Partial validation never updates
best-model selection or writes a completed-epoch metric.** The next session
repeats that whole epoch. There is no mid-epoch resume or emergency checkpoint
that falsely marks incomplete work as complete.

An epoch-zero `last.pt` is written before training, so even a stop during the
first epoch has a restart state. It is untrained and rejected by `infer.py` for
evaluation. The notebook handles empty metric history without failing. A
time-budget stop writes `session_stop.json` with its reason, elapsed hours,
last completed epoch, and resume path. It is a successful script return so the
remaining notebook cells can finish.

This is a cooperative guard, **not an operating-system timeout**. It cannot
interrupt an in-flight GPU kernel, blocked image read, checkpoint write, or
other stuck operation, and can overshoot by that operation's duration. It
cannot prevent external cancellation, GPU quota exhaustion, or service failure.
Epoch count alone also cannot guarantee runtime. The six-epoch cap and time
guard address different sources of uncertainty.

## Checkpoints, saving outputs, and continuation

| Artifact | Meaning |
| --- | --- |
| `last.pt` | Epoch-zero restart state initially; latest completed epoch afterward |
| `best_joint.pt` | Highest validation RGB PSNR among completed joint epochs |
| `best.pt` | Best checkpoint from the final configured phase; joint in default E2E |
| `best_gray.pt`, `best_color.pt` | Only created if those phases run and validate |
| `config.yaml`, `split.json` | Actual experiment settings and exact split IDs |
| `metrics.jsonl` | Completed-epoch history |
| `environment.json` | Python/PyTorch version, device, parameter and sample counts |
| `source_revision.txt` | Git revision recorded by the Kaggle notebook |
| `session_stop.json` | Time-limit stop details, if triggered |

`last.pt` is updated **after all training batches and scheduled validation for
an epoch**, not every batch. Logs confirm the completed-epoch save explicitly.
Atomic saves write a temporary file then replace the final file; this protects
the preceding local checkpoint from a partial replacement, not from losing the
entire runtime filesystem. With `val_every > 1`, some complete training epochs
have no validation; best models change only on validated epochs.

Full state contains model parameters, optimizer, AMP scaler, completed epoch,
phase, best phase scores, config, split IDs, and Python/NumPy/PyTorch/CUDA and
training-loader RNG states. LR is derived from the saved epoch and configured
schedule rather than a separately advanced scheduler. Exact CPU continuation
is tested; bitwise CUDA equality across environments is not guaranteed.

[Kaggle's notebook documentation](https://www.kaggle.com/docs/notebooks)
describes Save & Run All as a separate background session and saved notebook
versions as snapshots containing output files. Closing your browser does not
require stopping the commit. Checkpoint writes inside `/kaggle/working` are
**not continuous durable backups**. Letting the notebook finish normally is the
intended output-publication path; don't rely on cancellation, hard timeout,
or infrastructure failure to publish files. The guard reduces that risk but
does not guarantee recovery from those failures.

To continue an unfinished six-epoch run:

1. Save the normally completed notebook output and attach it to a new session.
2. Set `RESUME` to its mounted `runs/sar_gray_color/last.pt`.
3. Leave the mode and epoch schedule unchanged; the notebook reloads them from
   the checkpoint. Set `MAX_TRAIN_HOURS` for the new session.
4. Run the updated notebook. If the saved epoch is 4, epochs 5–6 remain. A new
   eight-hour budget does not restart the LR schedule or add six more epochs.

Available best checkpoints and history in the mounted checkpoint's directory
are copied into a new output folder, excluding epochs later than the resume
point. This preserves prior best models even if the continuation finds no
improvement. Keep the previous output attached and resume from `last.pt`.
Use a new run directory when branching from an older checkpoint rather than
mixing histories from different resume points.

`STOP_AFTER_EPOCH=3` is an optional manual stopping point after absolute epoch
three, not three additional epochs. The default is None. A checkpoint saved at
the final configured epoch means the schedule is complete; resume adds no more
epochs. A fresh training call refuses to overwrite an existing `last.pt`.

**Resume vs initialization:** `RESUME` restores the full experiment and forbids
changes to mode, epoch lengths, architecture, crop/batch/optimizer/loss settings,
or split settings. Mount/output paths, device, loader worker count, and session
time budget may change. `INIT_WEIGHTS` loads only model weights into a new
schedule, resetting optimizer, LR, RNG, best scores, and epoch count. It requires
a compatible architecture. To switch the canceled hybrid experiment to the
new E2E schedule, start fresh, or use its checkpoint with `INIT_WEIGHTS` if you
actually recovered one. Do not use both controls at once.

## Before the next Kaggle commit

Push these changes to GitHub **and upload the updated training notebook**.
Updating YAML alone is insufficient if an older notebook still explicitly sets
`MODE="hybrid"`. Confirm the printed config has mode `e2e`, phase lengths
0/0/6, and `max_hours: 8.0`, followed by:

```text
Schedule: e2e; 6 epochs; test split is not used.
Session training budget: 8 hours, including validation.
```

Keep `RESUME=None` for the new experiment, set the repository/data paths, enable
a GPU and Internet, and use Save & Run All. The model uses one GPU even if two
T4s are listed. If changing crop, width, batch, or losses for memory/performance,
record those choices as a different experiment. Use the validation holdout for
those decisions and the testing notebook only for final reporting.
