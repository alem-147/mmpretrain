_base_ = [
    '../_base_/models/snakeswin/tiny_224.py',
    '../_base_/datasets/imagenet_bs128_snakeswin_224.py',
    '../_base_/schedules/imagenet_bs128_adamw_snakeswin.py',
    '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
