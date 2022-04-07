# CheXpert: Automated Chest X-ray interpretation


## Summary: 
In this project, we will combine approaches from previous machine learning models to understand how model performance can be improved by accounting for the label uncertainty and utilizing novel loss functions.

## Approach: 
We will first reproduce the baseline DenseNet model \cite{densenet} outlined in by \citeauthor{pham2021interpreting}, in which the authors train a model ignoring the uncertain labels (\textif{U-ignore}) \cite{pham2021interpreting}. We will then improve the model performance by implementing various combinations of the methods that are used to account for label uncertainty. This methodology used by \citeauthor{pham2021interpreting} is currently second on the CheXpert leaderboard. 

We will attempt to further improve on the the work by \citeauthor{pham2021interpreting} by implementing Deep AUC Maximization instead of using cross entropy loss. Deep AUC Maximization has been demonstrated to perform well on the CheXpert data set and is currently in first place on the CheXpert leaderboard \cite{deepAUC}.

- Leaderboard: https://stanfordmlgroup.github.io/competitions/chexpert/
- Dataset: https://stanfordmlgroup.github.io/competitions/chexpert/. If you are lazy to sign up:
  - 11gb downsampled data: https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&h=370034d352dd1628094f9a95308d7fc58886539216630a859ae552e93064f576&v=1&xid=d222edbc02&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed
  - Full 439 gb data: https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0.zip&h=e97eea94025615b3d6b71fb0c528af7b27a7ded0e616f4c527d538fe342cdac9&v=1&xid=d222edbc02&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed

## A collective links of resources:
- Vietnamese VinAI paper (2nd on leaderboard): https://arxiv.org/abs/1911.06475
  - (From the same author) label smoothing technique they do to up the accuracy score: https://huyhieupham.github.io/data/vinbid_chexpert_solution_presentation.pdf
- A specific library for working with chest X-ray datasets and deep learning models, common interface and common pre-processing chain for a wide set of publicly available chest X-ray datasets => we can learn what kind of common preprocessing, or other good stuff here. Github: https://github.com/mlmed/torchxrayvision, paper: https://arxiv.org/abs/2111.00595
- DenseNet121 is the common pre-trained model for X-ray, but it’s a heavy model, requiring a buffy GPU to train. There’s a paper claiming that smaller networks such as Resnet 34 or even VGG 16 can classify radiographs as precisely as Densenet https://arxiv.org/pdf/2002.08991.pdf. 
- There’s an interesting medium posts (https://towardsdatascience.com/does-imagenet-pretraining-work-for-chest-radiography-images-covid-19-2e2d9f5f0875) (which included few interesting papers) about using ImageNet pretrained model then finetuning on a covid xray dataset, which has a better result than creating and training a model from scratch (COVID-NET, https://arxiv.org/pdf/2003.09871.pdf). We can try to use the CheXpert dataset as a bridge between Imagenet and Chest Radiography images by fine-tuning Imagenet models on CheXpert dataset and then apply to the problem at hand (Covid Lung detection)
-  If we need to remove unnecessary features from the X-ray and only keep the important area (e.g. for lung diseases, we only keep the lung area), then we can use something like UNet to segment out the lung area, then perform classification; similar to this: https://github.com/limingwu8/Lung-Segmentation, which is used for this kaggle competition task: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge. There’s also another dataset of chest Xray by VinBigData (a research lab from Vietnam) (https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview).
