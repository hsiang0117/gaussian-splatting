# Research fork: image loading and static relighting control

The Gaussian model, SH appearance, renderer and CUDA extension sources retain
their original behavior. This fork uses on-demand PIL image loading, a shared
CPU pixel cache, bounded float lifetimes and two-frame CUDA prefetch. Dataset
construction reads image headers only. RGB, RGBA alpha, resize and exposure-half
mask values match the previous loader exactly. The random camera sampler still
draws without replacement each cycle.

`--data_device cpu` is the default. `--image_cache_max 0` caches all decoded
frames; a positive value bounds the LRU cache by frame count. For 1460 RGBA
1024×1024 PNGs, the decoded uint8 payload is 5.70 GiB, excluding active frames,
model and process overhead. No CUDA rebuild is required for these Python edits.

Output defaults to `output/YYYYMMDD_HHMMSS`. `--preview_interval 1000` saves a
fixed training camera every 1000 steps, plus GT and camera metadata in
`rendertest`. Training configuration, step statistics and a final summary are
saved alongside the model. Relative source paths prevent duplicate frame names
when camXX directories contain the same image stem.

## Env-on static control

Use the same 200000-point cloud as the other cloud methods. Verify
`points3d.ply` before training: the fallback Blender initialization is a small
random cube and is inappropriate for this dataset. Use the supplied 1308/152
split, 1024×1024 images, black background, seed 0 and default 30000 steps.
Per-image exposure fitting remains disabled.

```bash
.venv/bin/python train.py -s /myfiles/data/CloudDatasetUniform_envon --eval --resolution 1 --data_device cpu --disable_viewer --preview_interval 1000 --checkpoint_iterations 10000 20000 30000
.venv/bin/python render.py -m output/<run> --skip_train
.venv/bin/python metrics.py -m output/<run>
.venv/bin/python tools/eval_test_groups.py output/<run>
.venv/bin/python tools/eval_static_lighting.py output/<run>
```

Sun direction, camera ID and sun/time ID are metadata only. They are never fed
to the original static renderer. `manifest.json` links rendered frames to
source images; `grouped_results.json` separates the 96 held-out-sun frames from
the 56 seen-sun/new-combination frames. Metrics stream one image pair at a time
and use the same VGG-LPIPS protocol as the sun-conditioned baseline.

`lighting_response.json` checks predictions at identical camera poses under
different target sun directions. It also reports GT changes for a deterministic
representative pair, selected by camera identity and sun-angle separation rather
than reconstruction error. A static model may train successfully and produce a
recognizable average appearance while remaining unable to reproduce changing
illumination. Report the measured outcome; optimization failure is not assumed.

## Verification

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Checks cover exact PIL/RGB/alpha values, CUDA-prefetched loss and gradient
agreement, cache reuse and eviction, full-queue sampling, worker errors and
shutdown, header-only dataset reads, unique frame IDs, metric coverage and
lighting response. The static renderer check verifies that recorded sun
metadata changes neither rendered pixels nor geometry gradients at a fixed pose.
