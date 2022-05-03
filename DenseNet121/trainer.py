import libauc.losses
import libauc.optimizers
import libauc.models
import torch 
# from PIL import Image
import numpy as np
from tqdm import tqdm
import AUCMMS
import pesgg
from fastai.vision.all import *

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def get_densenet_model(seed,num_classes=5):
    set_all_seeds(seed)
    model = libauc.models.DenseNet121(pretrained=True, last_activation=None, activations='relu', num_classes=num_classes)
    model = model.cuda()
    return model


def get_loss_and_optimizer(model,lr,loss_type='bce',num_classes=5,imratio=None,use_fastai=True,
                           weight_decay=0,gamma=500,margin=1.0):
    loss,opt=None,None
    if loss_type =='bce':
        loss = BCEWithLogitsLossFlat()
        if not use_fastai:
            opt = libauc.optimizers.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = partial(OptimWrapper,opt = libauc.optimizers.Adam,lr=lr,weight_decay=weight_decay)
            
    elif loss_type=='aucm':
        loss = AUCMMS.AUCM_MultiLabel_Sigmoid(imratio=imratio,num_classes=num_classes)
        if not use_fastai:
            opt = libauc.optimizers.PESG(model, 
                 a=loss.a, 
                 b=loss.b, 
                 alpha=loss.alpha, 
                 imratio=imratio, 
                 lr=lr, 
                 gamma=gamma, 
                 margin=margin, 
                 weight_decay=weight_decay,device='cuda')
        else:
            opt = partial(OptimWrapper,opt = pesgg.PESGG,
                   model=model,
                   a=loss.a,b=loss.b,
                   alpha=loss.alpha,
                   imratio = imratio,
                   lr = lr,
                   gamma=gamma,
                   margin=margin,
                   weight_decay=weight_decay)
    return loss,opt

def train_fastai_learner(fastai_loader,model,loss,opt,lr,n_epochs,
                         monitor='roc_auc_score',
                         weight_name='weight',
                         lr_scheduler='1cycle', # either '1cycle','flat_cos' or 'normal'
                         use_fp16=True,
                         mixup_alpha=None):
    #define callback
    cbs = [
              ShowGraphCallback(),
              SaveModelCallback(monitor=monitor,fname=weight_name, comp=np.greater if monitor=='roc_auc_score' else np.less)
          ]
    if mixup_alpha is not None:
        mixup = MixUp(mixup_alpha)
        cbs.append(mixup)
        
    fastai_learn = Learner(fastai_loader,model,loss_func = loss,
                           opt_func = opt,
                           metrics=[RocAucMulti(average='macro'),
                                    RocAucMulti(average=None),
                                    ],
                          model_dir = 'model_weights',
                          cbs=cbs
                          )
    if use_fp16:
        fastai_learn = fastai_learn.to_fp16()
    
    if lr_scheduler=='1cycle':
        fastai_learn.fit_one_cycle(n_epochs,lr_max = lr)
    elif lr_scheduler=='flat_cos':
        fastai_learn.fit_flat_cos(n_epochs,lr = lr,pct_start=0.2)
    else:
        fastai_learn.fit(n_epochs,lr = lr)