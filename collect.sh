 apptainer exec --nv --cleanenv \
    --env VPT_RENDER_BACKEND=virtualgl \
    --env VGL_DISPLAY=egl0 \
    --bind /data/scene-rep/u/echen/Video-Pre-Training:/outputs \
    vpt-dev.sif \
    /app/docker/vpt-minerl-entrypoint.sh \
    python /app/collect_policy.py \
      --model /app/checkpoints/foundation-model-1x.model \
      --weights /app/checkpoints/foundation-model-1x.weights \
      --out-dir /outputs \
      --episodes 1 \
      --max-steps 12000 \
      --save-video
