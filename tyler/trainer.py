import numpy as np

import torch
import torch.nn as nn

from transformers import Trainer

from libauc.losses import AUCM_MultiLabel
from libauc.optimizers import PESG

class MultiLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

class AUCTrainer(Trainer):
    def __init__(self, *args, imratio=None, margin=1.0, gamma=500, **kwargs):
        if imratio is None:
            imratio = [0.1] * 5
        self._AUC_loss = AUCM_MultiLabel(imratio=imratio, num_classes=len(imratio), margin=margin)
        self._AUC_margin = margin
        self._AUC_gamma = gamma
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        self.optimizer = PESG(
            self.model,
            a=self._AUC_loss.a,
            b=self._AUC_loss.b,
            alpha=self._AUC_loss.alpha,
            lr=self.args.learning_rate,
            gamma=self._AUC_gamma,
            margin=self._AUC_margin,
            weight_decay=self.args.weight_decay
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        sigmoid = torch.nn.Sigmoid()

        loss = self._AUC_loss(sigmoid(logits), labels)
        return (loss, outputs) if return_outputs else loss
