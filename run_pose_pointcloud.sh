torchrun --nproc_per_node=1 run_pose_pointcloud.py \
    --test_dataset "1 @ Own(split='train', ROOT='/data0/zsz/mast3recon/data/test_own2', resolution=(512,384), seed=19417, num_views=8, gt_num_image=0, aug_portrait_or_landscape=False, sequential_input=False)" \
    --model "AsymmetricMASt3R(wogs=True, low_res=True, ft32=True, inject_lowtoken=True,  pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --pretrained "/data0/zsz/mast3recon/checkpoints/dust3r_demo_512dpt_clean_64gpu_SA_v2_all_v2_inject_3dv/checkpoint-60.pth" \
    --test_criterion "MeshOutput(sam=False)" \
    --lr 1e-04 --min_lr 1e-05 --warmup_epochs 1 --epochs 100 --batch_size 1 --accum_iter 1 \
    --save_freq 1 --keep_freq 5 --eval_freq 1 --print_freq=10 \
    --output_dir "checkpoints/dust3r_demo_512dpt_clean_32gpu_gssssdfd" --amp 1 --seed 123 --num_workers 0