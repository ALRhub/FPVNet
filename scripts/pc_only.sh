python run.py --config-name robocasa_fpv_config.yaml \
    --multirun agent_name=pc_only \
    agents/obs_encoders=point_mlp \
    obs_tokens=256 \
    if_film_condition=False \
    n_embd=128 \
    env_group=pnp1,pnp2,doors,drawer,stove,coffee,buttons,sink \
    seed=0,1,2