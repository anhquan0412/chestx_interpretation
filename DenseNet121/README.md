# Resnet121 experiments

Here are all the experiments done for the report, and an extra notebook about applying mixup as regularization toward Densenet121 base model

- [Uncertainty Replacement Experiments with 1cycle](1_pick_best_uncertainty_replacement.ipynb): where I ran experiments to compare different uncertainty replacement approach, such as U-Zeros, U-Ones, U-One+LSR (known as U-One smoothing in the notebook) and U-Default.
- [Two-stage method with 1cycle and progressive resizing](2_second_stage_AUCM_Multilabel_ProgressiveResizing.ipynb): where I ran two-stage methods to achieve highest AUC score possible for Chexpert 5-class multilabel task.

Deep Learning framework used:
- Pytorch: https://pytorch.org/
- Fastai: https://pypi.org/project/fastai/

Existing code and models to start:
- Using fastai dataloaders: https://docs.fast.ai/migrating_pytorch_verbose.html
- libauc's chexpert data import: https://github.com/Optimization-AI/ICCV2021_DeepAUC/blob/main/chexpert.py
- libauc library for DenseNet, Adam, PESG and AUCM loss : https://pypi.org/project/libauc/

Modification
- [dataloader.py](dataloader.py): based on libauc's chexpert.py, I added several label smoothing variation and sampling technique (so that I can perform several experiments on a fraction of the data to save times). I also added a dataloader helper to get the data loader for fastai/pytorch training
- [trainer.py](trainer.py): where I define helper functions to get optimizers or loss functions based on input parameters, and a trainer object to train the model with input params such as trainning with mixed precisions, what score to monitor, adding mixup augmentation or not, ...
- [AUCMMS.py](AUCMMS.py): originally from libauc library, with a small modification of adding sigmoid to y_pred before calculating the loss. This modification is needed for fastai learner
- [pesgg.py](PESGG.py): originally from libauc library, with a small modification of adding an input parameters ```params``` in the PESG optimizer, as this is a required input of any deep learning optimizer. This modification is needed for fastai learner

To run the notebooks, make sure to install appropriate packages mentioned in [experiment.yml](experiment.yml)



