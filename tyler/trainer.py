import numpy as np

import torch
import torch.nn as nn

from transformers import Trainer

class CETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, torch.argmax(labels, dim=-1))
        return (loss, outputs) if return_outputs else loss