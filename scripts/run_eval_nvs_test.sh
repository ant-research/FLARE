torchrun --nproc_per_node=1 eval_nvs.py \
    --test_dataset "Re10K(split='test', ROOT='/nas3/zsz/re10k', resolution=[(256, 256)], aug_crop='auto', aug_monocular=0, num_views=2, gt_num_image=3, aug_portrait_or_landscape=False, meta='assets/evaluation_index_re10k.json')" \
    --model "AsymmetricMASt3R_2v(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --test_criterion "Eval_NVS(scaling_mode='precomp_5e-4_0.3_4', alpha=0.2)" \
    --lr 4e-05 --min_lr 1e-05 --warmup_epochs 1 --epochs 100 --batch_size 1 --accum_iter 1 \
    --save_freq 1 --keep_freq 25 --eval_freq 1 --print_freq=200 --pretrained "/nas3/zsz/h20_checkpoint/DL3DV_low_maxgap22_hyb_min_gap_45_max_gap_110/checkpoint-best.pth" --amp 0 --seed 0 --num_workers 0

