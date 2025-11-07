python run.py --config-name robocasa_fpv_config.yaml \
    --multirun agent_name=fpv_sugar \
    agents/model=beso/beso_cond_dec_transformer \
    agents/obs_encoders=sugar_resnet_local_img_cond_transformer \
    obs_tokens=256 \
    if_film_condition=True \
    n_embd=256 \
    use_pc_color=True \
    normalize_neighborhood=True \
    env_group=pnp1,pnp2,doors,drawer,stove,coffee,buttons,sink \
    seed=0,1,2