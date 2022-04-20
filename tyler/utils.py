import numpy as np
import torch
from datasets import load_metric
from scipy.special import softmax

metric = load_metric("accuracy")
def compute_metrics(p):
    sm = softmax(p.predictions, axis=-1)
    return metric.compute(predictions=np.argmax(sm, axis=1), references=np.argmax(p.label_ids, axis=1))

def collate_fn(batch):
    retval = {}
    retval["pixel_values"] = torch.stack([x['pixel_values'] for x in batch])
    if batch[0].get('labels'):
        retval['labels'] =  torch.stack([x['labels'] for x in batch])
    return retval

    