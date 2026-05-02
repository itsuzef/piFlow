# FourierPolicy inference config — ImageNet 256x256.
#
# Mirrors `gmdit_k32_imagenet_piid_1step_test.py` with the GM-specific
# knobs replaced by the Fourier head (`use_fourier=True`, `num_harmonics=4`).
#
# Notes:
#   • Uses `GMDiTTransformer2DModel` (V1). Porting to V2 is deferred
#     until V2 becomes the canonical class.
#   • `pretrained=None` — no Fourier-trained checkpoint exists yet;
#     set to a checkpoint path once one is available.
#   • `denoising_mean_mode='U'` is the parent default and is only
#     consumed by `loss()`, which is unreachable under `inference_only=True`.
#     The Fourier head outputs `x_hat_0` + `fourier_sin_coeffs` directly
#     (FourierModelOutput); no u→x_0 conversion is required.

name = 'piflow_fourier_inference'

model = dict(
    type='LatentDiffusionClassImage',
    vae=dict(
        type='PretrainedVAE',
        model_name_or_path='stabilityai/sd-vae-ft-ema',
        freeze=True,
        torch_dtype='float16'),
    diffusion=dict(
        type='PiFlowImitation',
        policy_type='Fourier',
        policy_kwargs=dict(checkpointing=True),
        denoising=dict(
            type='GMDiTTransformer2DModel',
            pretrained=None,
            use_fourier=True,
            num_harmonics=4,
            num_attention_heads=16,
            attention_head_dim=72,
            in_channels=4,
            num_layers=28,
            sample_size=32,  # 256
            torch_dtype='bfloat16'),
        num_timesteps=1000,
        timestep_sampler=dict(type='ContinuousTimeStepSampler', shift=1.0, logit_normal_enable=False),
        denoising_mean_mode='U'),
    diffusion_use_ema=True,
    inference_only=True
)

work_dir = f'work_dirs/{name}'
# yapf: disable
train_cfg = dict()
test_cfg = dict(
    nfe=1,
    total_substeps=128,
)

data = dict(
    workers_per_gpu=4,
    val=dict(
        type='ImageNet',
        data_root='data/imagenet/train_cache/',
        datalist_path='data/imagenet/train_cache.txt',
        negative_label=1000,
        latent_size=(4, 32, 32),
        test_mode=True),
    test_dataloader=dict(samples_per_gpu=125),
    persistent_workers=True,
    prefetch_factor=64)

prefix = 'step1'
evaluation = [
    dict(
        type='GenerativeEvalHook',
        data='val',
        prefix=prefix,
        feed_batch_size=32,
        viz_num=256,
        metrics=[
            dict(
                type='InceptionMetrics',
                num_images=50000,
                resize=False,
                reference_pkl='huggingface://Lakonik/inception_feats/imagenet256_inception_adm.pkl'),
            ],
        viz_dir=f'viz/{name}/{prefix}',
        save_best_ckpt=False)]

# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
mp_start_method = 'fork'
