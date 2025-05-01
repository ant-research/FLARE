export CUDA_VISIBLE_DEVICE=2,
torchrun --nproc_per_node=1 --master_port=2122 test_pose_10k.py --test_dataset "Re10K_pose(split='test', ROOT='/nas3/zsz/RealEstate10k_v2', resolution=(512,384), seed=772343247, num_views=5, gt_num_image=0, aug_portrait_or_landscape=False)" \
--model "AsymmetricMASt3R(wpose=False, pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
--test_criterion "IMC_10k(L21, norm_mode='avg_dis')" \
--pretrained "/nas3/zsz/FLARE_clean/checkpoints/geometry_pose.pth" \
--lr 1e-05 --min_lr 1e-05 --warmup_epochs 1 --epochs 100 --batch_size 1 --accum_iter 8 \
--save_freq 1 --keep_freq 5 --eval_freq 1 \
--output_dir "checkpoints/re10k" --seed 2  --noise_rot 0  --noise_trans 0. --noise_prob 0.5 --num_workers 0 --amp 1 
