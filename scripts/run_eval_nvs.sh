torchrun --nproc_per_node=1 eval_nvs.py \
    --test_dataset "Re10K(split='test', ROOT='/nas3/zsz/re10k', resolution=[(256, 256)], aug_crop='auto', aug_monocular=0, num_views=2, gt_num_image=3, aug_portrait_or_landscape=False, meta='assets/evaluation_index_re10k.json')" \
    --model "AsymmetricMASt3R_2v(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --test_criterion "Eval_NVS(scaling_mode='precomp_5e-4_0.3_4', alpha=0.2)" \
    --pretrained "checkpoints/NVS.pth" --amp 0 --num_workers 0

