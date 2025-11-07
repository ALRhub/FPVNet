python run.py --config-name robocasa_fpv_config.yaml \
    --multirun agent_name=pc+rgb \
    agents/obs_encoders=point_mlp_resnet \
    obs_tokens=259 \
    if_film_condition=True \
    n_embd=128 \
    env_group=pnp1,pnp2,doors,drawer,stove,coffee,buttons,sink \
    seed=0,1,2