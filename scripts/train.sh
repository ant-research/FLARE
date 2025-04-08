
torchrun --nproc_per_node=1 train.py \
    --train_dataset "12800 @ Co3d(mask_bg='rand',split='train', ROOT='/nas3/zsz/datasets/co3d_processed_full.zip', resolution=[(512, 384)], aug_crop='auto', aug_monocular=0.005,  num_views=8, gt_num_image=0, sequential_input=True, aug_portrait_or_landscape=False)" \
    --test_dataset "16 @ Co3d(mask_bg='rand',split='train', ROOT='/nas3/zsz/datasets/co3d_processed_full.zip', resolution=[(512, 384)], aug_crop='auto', aug_monocular=0.005,  num_views=8, gt_num_image=0, sequential_input=True, aug_portrait_or_landscape=False)" \
    --model "AsymmetricMASt3R(is_train=True, wpose=False, pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed', img_size=(512, 512), head_type='catmlp+dpt', output_mode='pts3d+desc24', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12, two_confs=True, desc_conf_mode=('exp', 0, inf))" \
    --train_criterion "ConfLoss(Regr3D_clean(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --test_criterion "ConfLoss(Regr3D_clean(L21, norm_mode='avg_dis'), alpha=0.2)" \
    --pretrained "/nas3/zsz/FLARE_clean/checkpoints/geometry_pose.pth" \
    --lr 1e-04 --min_lr 1e-05 --warmup_epochs 1 --epochs 300 --batch_size 2 --accum_iter 1 \
    --save_freq 1 --keep_freq 25 --eval_freq 1 --print_freq=200 --disable_cudnn_benchmark \
    --output_dir "output/finetune" --amp 1 --num_workers 16 --gt_num_image 0 --cycle_epoch 100 

