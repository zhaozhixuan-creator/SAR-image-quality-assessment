_base_ = './pix2pix_vanilla_unet_bn_x2ka_b1x1_220k.py'

model = dict(
    gen_auxiliary_loss=[
        dict(
            type='L1Loss',
            loss_weight=50.0,
            loss_name='pixel_loss',
            n_edge=32,
            data_info=dict(pred='fake_ka', target='real_ka'),
            reduction='mean'),
        dict(
            type='pdfLoss',
            loss_type='kl',
            loss_weight=500.0,
            loss_name='loss_pdf_kl',
            n_edge=32,
            width_patch=256,
            n_bins=64,
            value_range=(-1.0, 1.0),
            data_info=dict(pred='fake_ka', target='real_ka'),
            reduction='mean')
    ])
