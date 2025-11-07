python run.py --config-name robocasa_fpv_config.yaml \
    --multirun agent_name=rgb_only \
    agents/obs_encoders=pretrained_resnet_film \
    obs_tokens=3 \
    if_film_condition=True \
    n_embd=256 \
    env_group=pnp1,pnp2,doors,drawer,stove,coffee,buttons,sink \
    seed=0,1,2