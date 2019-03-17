from .mask_mle_task import MaskMLETask
from fairseq.tasks import register_task


@register_task("mask_gan")
class MaskGANTask(MaskMLETask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
