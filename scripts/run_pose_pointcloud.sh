torchrun --nproc_per_node=1 run_pose_pointcloud.py \
    --test_dataset "1 @ CustomDataset(split='train', ROOT='assets/room', resolution=(512,384), seed=1, num_views=6, gt_num_image=0, aug_portrait_or_landscape=False, sequential_input=False, wpose=False)" \
    --model "AsymmetricMASt3R(wpose=False, pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --pretrained "checkpoints/geometry_pose.pth" \
    --test_criterion "MeshOutput(sam=False)" --output_dir "log/" --amp 1 --seed 1 --num_workers 0
