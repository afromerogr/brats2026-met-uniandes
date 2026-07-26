import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerRCOversample(nnUNetTrainer):
    """ResEnc-L + foreground oversampling 0.90 + RC (label 4) CE upweight 3x."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.oversample_foreground_percent = 0.90

    def _build_loss(self):
        base = super()._build_loss()
        tgt = base.loss if isinstance(base, DeepSupervisionWrapper) else base
        ce = getattr(tgt, 'ce', None)
        if ce is not None:
            n = self.label_manager.num_segmentation_heads
            w = torch.ones(n, device=self.device)
            if n > 4:
                w[4] = 3.0
            ce.weight = w
        return base
