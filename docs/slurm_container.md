# VPT MineRL Containers on Slurm

This repo includes a headless collector for running a VPT policy in MineRL on
Slurm nodes. Minecraft/Malmo still needs an X server, but plain Xvfb often uses
CPU OpenGL and can slow MineRL down substantially. The container therefore
installs VirtualGL and defaults GPU jobs to `vglrun -d egl` with a private Xvfb
display. Plain Xvfb is kept as a CPU-only fallback.

The rendering backend is controlled by `VPT_RENDER_BACKEND`:

- `auto`: use VirtualGL when a GPU device is visible, otherwise fall back.
- `virtualgl`: force `vglrun -d "$VGL_DISPLAY"`; default `VGL_DISPLAY=egl`.
- `xvfb`: force CPU-rendered Xvfb.
- `native`: use the existing `DISPLAY` without wrapping.

## Build Docker

Build from the repository root:

```bash
docker build -t vpt-minerl:latest .
```

Run locally with a GPU:

```bash
docker run --rm --gpus all --device /dev/dri \
  -e VPT_RENDER_BACKEND=virtualgl \
  -e VGL_DISPLAY=egl \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/rollouts:/outputs" \
  vpt-minerl:latest \
  python /app/collect_policy.py \
    --model /models/foundation-model-1x.model \
    --weights /models/foundation-model-1x.weights \
    --out-dir /outputs \
    --episodes 1 \
    --max-steps 12000 \
    --save-video
```

For CPU-only smoke tests, omit `--gpus all` and pass `--device cpu`.
Also set `-e VPT_RENDER_BACKEND=xvfb` for those CPU runs.

Verify GPU OpenGL before a long run:

```bash
docker run --rm --gpus all --device /dev/dri \
  -e VPT_RENDER_BACKEND=virtualgl \
  vpt-minerl:latest \
  glxinfo -B
```

The `OpenGL renderer` line should name an NVIDIA GPU. If it says `llvmpipe`,
you are still rendering on CPU.

## Build Apptainer

Build directly from the Apptainer definition:

```bash
apptainer build vpt-minerl.sif apptainer/vpt-minerl.def
```

If your cluster only allows unprivileged builds through a remote builder, use the
cluster's documented `apptainer build --remote` flow with the same definition.
Direct Apptainer builds copy the whole working tree before `%post`; building from
the Docker image first can be faster because `.dockerignore` excludes large local
artifacts.

You can also build the Docker image first and convert it:

```bash
docker build -t vpt-minerl:latest .
apptainer build vpt-minerl.sif docker-daemon://vpt-minerl:latest
```

Verify GPU OpenGL:

```bash
apptainer exec --nv \
  --env VPT_RENDER_BACKEND=virtualgl \
  vpt-minerl.sif \
  /app/docker/vpt-minerl-entrypoint.sh \
  glxinfo -B
```

## Submit a Slurm Collection Job

Set paths on the submit host, then submit:

```bash
mkdir -p logs rollouts
sbatch \
  --export=ALL,IMAGE="$PWD/vpt-minerl.sif",MODEL="$PWD/models/foundation-model-1x.model",WEIGHTS="$PWD/models/foundation-model-1x.weights",EPISODES=1,MAX_STEPS=12000,COLLECT_EXTRA_ARGS="--save-video" \
  slurm/collect_vpt.sbatch
```

The Slurm script bind-mounts the model directory read-only and writes outputs to
`$OUT_DIR`, defaulting to `rollouts/$SLURM_JOB_ID`.

Useful environment overrides:

- `OUT_DIR=/shared/path`: output directory on the host.
- `APPTAINER_NV=0`: disable `apptainer exec --nv` for CPU-only nodes.
- `RENDER_BACKEND=xvfb`: force the slower CPU-rendering fallback.
- `VGL_DISPLAY=egl`: VirtualGL device selector; use `/dev/dri/cardN` or `eglN`
  when your cluster needs an explicit render device.
- `COLLECT_EXTRA_ARGS="--device cpu --seed 123 --log-every 500"`: extra collector flags.

## Output Layout

Each run creates a run directory under `--out-dir`:

```text
run_metadata.json
summary.json
episode_00000/
  actions.jsonl
  summary.json
  pov.mp4        # only when --save-video is used
```

`actions.jsonl` contains one JSON object per environment step with the MineRL
action, reward, termination flags, and info payload. Video frame 0 is the reset
observation; action step `N` maps video frame `N` to frame `N+1`.

## Notes

- The VirtualGL/EGL path follows the MineRL performance-tip guidance and the
  headless GPU-rendering pattern used by the referenced `egl-docker` images.
- VirtualGL 3.1.1 and newer are published on GitHub rather than SourceForge, so
  the image installs the pinned GitHub release asset with a SHA-256 check.
- The image builds the vendored MineRL/Malmo client during `uv sync`, so the
  build host needs network access for Python packages, Gradle artifacts, and the
  MCP-Reborn checkout.
- Runtime jobs should not need network access.
- Keep large `.model`, `.weights`, and rollout artifacts outside the image and
  bind them into Docker or Apptainer at runtime.

## Troubleshooting

If PyTorch aborts with a message like `Unable to load any of
{libcudnn_graph.so...}` under `apptainer exec --nv`, host CUDA/cuDNN paths are
probably taking precedence over the CUDA libraries installed with the PyTorch
wheel. The entrypoint prepends `/app/.venv/lib/python*/site-packages/nvidia/*/lib`
to `LD_LIBRARY_PATH` before launching the collector, so rebuild the SIF after
entrypoint changes and prefer `apptainer exec --nv --cleanenv`.

If VirtualGL prints `failed to open /dev/dri/cardN: Permission denied`, the job
can see a DRM render node that your user or Slurm device cgroup cannot open.
Try selecting the allocated EGL device explicitly, for example
`--env VGL_DISPLAY=egl0` or `--env VGL_DISPLAY=egl1`, and verify with
`glxinfo -B`. If every device fails, the cluster needs to grant access to the
allocated GPU's `/dev/dri` render node, or you must fall back to
`RENDER_BACKEND=xvfb`.
