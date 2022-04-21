import numpy as np
import torch
from datasets import load_metric
from scipy.special import expit

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataloader import COMPETITION_TASKS

def compute_metrics(p):
    threshold = 0.5
    metrics = {}

    sigmoids = expit(p.predictions)
    metrics["accuracy"] = accuracy_score(np.round(p.label_ids), (sigmoids >= threshold).astype("int"))
    metrics["f1"] = f1_score(np.round(p.label_ids), (sigmoids >= threshold).astype("int"), average="micro")

    for i, task in enumerate(COMPETITION_TASKS):
        metrics[f"AUC_{task}"] = roc_auc_score(np.round(p.label_ids[:, i]), sigmoids[:, i])

    return metrics



def collate_fn(batch):
    retval = {}
    retval["pixel_values"] = torch.stack([x['pixel_values'] for x in batch])
    if batch[0].get('labels') is not None:
        retval['labels'] =  torch.stack([x['labels'] for x in batch])
    return retval

    