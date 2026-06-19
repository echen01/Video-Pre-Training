.venv/bin/python lancedb_rollout.py \
    --model checkpoints/2x.model \
    --weights checkpoints/rl-from-early-game-2x.weights \
    --idm-model checkpoints/4x_idm.model \
    --idm-weights checkpoints/4x_idm.weights \
    --out-dir debug_shards \
    --episodes 1 \
    --max-steps 4000 \
    --device cuda \
    --render \
    --save-video
    #--render \