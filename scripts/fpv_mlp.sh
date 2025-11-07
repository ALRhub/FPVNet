python run.py --config-name robocasa_fpv_config.yaml \
    --multirun agent_name=fpv_mlp \
    agents/model=beso/beso_cond_dec_transformer \
    agents/obs_encoders=point_mlp_resnet_local8x8_img_cond_transformer \
    obs_tokens=256 \
    if_film_condition=True \
    n_embd=128 \
    latent_dim=512 \
    env_group=pnp1,pnp2,doors,drawer,stove,coffee,buttons,sink \
    seed=0,1,2