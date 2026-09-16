source_domain = 'x'
target_domain = 'ka'
input_channel = 6
output_channel = 6
n_edge_visual = 32
n_down = 6

model = dict(
    type='Pix2Pix',
    generator=dict(
        type='UnetGenerator',
        in_channels=input_channel,
        out_channels=output_channel,
        num_down=n_down,
        base_channels=64,
        norm_cfg=dict(type='BN'),
        use_dropout=True,
        init_cfg=dict(type='normal', gain=0.001)),
    discriminator=dict(
        type='PatchDiscriminator',
        in_channels=input_channel + output_channel,
        base_channels=64,
        num_conv=3,
        norm_cfg=dict(type='BN'),
        init_cfg=dict(type='normal', gain=0.001)),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        real_label_val=1.0,
        fake_label_val=0.0,
        loss_weight=1.0),
    default_domain=target_domain,
    reachable_domains=[target_domain],
    related_domains=[target_domain, source_domain],
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=50.0,
            loss_name='pixel_loss',
            n_edge=n_edge_visual,
            data_info=dict(
                pred=f'fake_{target_domain}',
                target=f'real_{target_domain}'),
            reduction='mean'),
        dict(
            type='pdfLoss',
            loss_weight=500.0,
            loss_name='pixel_loss',
            n_edge=n_edge_visual,
            width_patch=256,
            data_info=dict(
                pred=f'fake_{target_domain}',
                target=f'real_{target_domain}'),
            reduction='mean')
    ])
train_cfg = None
test_cfg = None

resume_from = 'https://download.openmmlab.com/mmgen/pix2pix/refactor/pix2pix_vanilla_unet_bn_1x1_80k_facades_20210902_170442-c0958d50.pth'

img_norm_cfg = dict(
    mean=dict(
        img_x=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        img_ka=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    std=dict(
        img_x=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        img_ka=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))

train_pipeline = [
    dict(
        type='LoadPairedPolSARFromFile',
        key='pair',
        domain_a=source_domain,
        domain_b=target_domain),
    dict(
        type='FlipSAR',
        keys=[f'img_{source_domain}', f'img_{target_domain}'],
        direction='horizontal'),
    dict(
        type='NormalizeSAR',
        keys=[f'img_{source_domain}', f'img_{target_domain}'],
        use_log=False,
        **img_norm_cfg),
    dict(
        type='ImageToTensor',
        keys=[f'img_{source_domain}', f'img_{target_domain}']),
    dict(
        type='Collect',
        keys=[f'img_{source_domain}', f'img_{target_domain}'],
        meta_keys=[f'img_{source_domain}_path', f'img_{target_domain}_path'])
]
test_pipeline = [
    dict(
        type='LoadPairedPolSARFromFile',
        key='pair',
        domain_a=source_domain,
        domain_b=target_domain),
    dict(
        type='NormalizeSAR',
        keys=[f'img_{source_domain}', f'img_{target_domain}'],
        use_log=False,
        **img_norm_cfg),
    dict(
        type='ImageToTensor',
        keys=[f'img_{source_domain}', f'img_{target_domain}']),
    dict(
        type='Collect',
        keys=[f'img_{source_domain}', f'img_{target_domain}'],
        meta_keys=[f'img_{source_domain}_path', f'img_{target_domain}_path'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type='PairedImageDataset',
        dataroot='data/x2ka_aircas/paired',
        pipeline=train_pipeline,
        test_mode=False),
    test=dict(
        type='PairedImageDataset',
        dataroot='data/wholeImg',
        pipeline=test_pipeline,
        test_mode=True,
        testdir='val'))

optimizer = dict(
    generators=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)),
    discriminators=dict(type='Adam', lr=2e-4, betas=(0.5, 0.999)))
lr_config = None
checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(
        type='MMGenVisualizationSARHook',
        output_dir='training_samples',
        res_name_list=[
            f'real_{source_domain}', f'fake_{target_domain}',
            f'real_{target_domain}'
        ],
        interval=1000,
        n_channel=output_channel,
        n_edge=n_edge_visual)
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
workflow = [('train', 1)]
total_iters = 220000
use_ddp_wrapper = True
find_unused_parameters = True
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'
