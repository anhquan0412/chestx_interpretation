import numpy as np
import torch 
from torch.utils.data import Dataset
import torchvision.transforms as tfs
import cv2
from PIL import Image
import pandas as pd

COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]

class CheXpertExtended(Dataset):
    '''
    Reference: 
        @inproceedings{yuan2021robust,
            title={Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
            author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
            booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
            year={2021}
            }
    '''
    def __init__(self, 
                 csv_path, 
                 image_root_path='',
                 image_size=224,
                 class_index=0,
                 sample_frac=1.0,
                 sample_seed=123,
                 uncertainty_method='default',
                 smooth_lower=0,
                 smooth_upper=1,
                 use_frontal=True,
                 use_upsampling=True,
                 flip_label=False,
                 shuffle=True,
                 verbose=True,
                #  upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 upsampling_cols=['Cardiomegaly'],
                 train_cols=COMPETITION_TASKS,
                 transforms=[tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05), scale=(0.95, 1.05), fill=128)],
                 mode='train'):
        
        self.transforms = transforms

        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')


        if use_frontal:
            self.df = self.df[self.df['Frontal/Lateral'] == 'Frontal']  
            
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print ('Upsampling %s...'%col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        self.df = self.df.copy() # to avoid slicing warning from df

        # impute missing values 
        if uncertainty_method=='default':
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']:
                    self.df[col].replace(-1, 1, inplace=True)  
                    # self.df[col].fillna(0, inplace=True) 
                elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']:
                    self.df[col].replace(-1, 0, inplace=True) 
                    # self.df[col].fillna(0, inplace=True)
                # else:
                    # self.df[col].fillna(0, inplace=True)
        elif uncertainty_method=='zero':
            for col in train_cols:
                self.df[col].replace(-1,0,inplace=True)
        elif uncertainty_method=='one':
            for col in train_cols:
                self.df[col].replace(-1,1,inplace=True)
        elif uncertainty_method=='smoothing': # U-one or U-zero smoothing
            for col in train_cols:
                self.df.loc[self.df[col] == -1, col] = np.random.uniform(smooth_lower, smooth_upper, size=self.df.loc[self.df[col] == -1, col].shape)
        elif uncertainty_method=='default-smoothing':
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']: # u-one smoothing
                    self.df.loc[self.df[col] == -1, col] = np.random.uniform(0.55, 0.85, size=self.df.loc[self.df[col] == -1, col].shape)  
                elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']: # u-zero smoothing
                    self.df.loc[self.df[col] == -1, col] = np.random.uniform(0.0, 0.3, size=self.df.loc[self.df[col] == -1, col].shape)  
                    
        elif uncertainty_method=='default-smoothing-one':
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']: # u-one smoothing
                    self.df.loc[self.df[col] == -1, col] = np.random.uniform(0.55, 0.85, size=self.df.loc[self.df[col] == -1, col].shape)  
                elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']: # u-zero
                    self.df[col].replace(-1, 0, inplace=True) 

        elif uncertainty_method=='default-smoothing-zero':
            for col in train_cols:
                if col in ['Edema', 'Atelectasis']: # u-one smoothing
                    self.df[col].replace(-1, 1, inplace=True)  
                elif col in ['Cardiomegaly','Consolidation',  'Pleural Effusion']: # u-zero
                    self.df.loc[self.df[col] == -1, col] = np.random.uniform(0.0, 0.3, size=self.df.loc[self.df[col] == -1, col].shape)  

                    
        for col in train_cols:
            self.df[col].fillna(0, inplace=True)

        # perform sampling (for experimenting)
        if sample_frac<1:
            self.df = self.df.sample(frac=sample_frac,random_state=sample_seed,ignore_index=True)

        self._num_images = len(self.df)
        
        # 0 --> -1
        if flip_label and class_index != -1: # In multi-class mode we disable this option!
            self.df.replace(0, -1, inplace=True)   
            
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(sample_seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
        
        
        assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1: # 5 classes
            print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:       # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()
        
        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size
        
        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()
    
        if class_index != -1:
            if verbose: print ('-'*30)
            if flip_label:
                self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                if verbose:
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
            else:
                self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                if verbose:
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                    print ('-'*30)
        else:
            if verbose: print ('-'*30)
            imratio_list = []
            for class_key, select_col in enumerate(train_cols):
                imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
                imratio_list.append(imratio)
                if verbose:
                    print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                    print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                    print ()
            self.imratio = np.mean(imratio_list)
            self.imratio_list = imratio_list
            if verbose: print ('-'*30)
            
    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)
       
    @property  
    def data_size(self):
        return self._num_images 
    
    def image_augmentation(self, image):
        if self.transforms is not None:
            img_aug = tfs.Compose(self.transforms)
            image = img_aug(image)
        return image
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = Image.fromarray(image)
        if self.mode == 'train':
            image = self.image_augmentation(image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)  
        image = image/255.0
        __mean__ = np.array([[[0.485, 0.456, 0.406]]])
        __std__ =  np.array([[[0.229, 0.224, 0.225]  ]]) 
        image = (image-__mean__)/__std__
        image = image.transpose((2, 0, 1)).astype(np.float32)
        if self.class_index != -1: # multi-class mode
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        else:
            label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        return image, label



def create_loaders(image_size,
                   sample_frac,
                   sample_seed,
                   uncertainty_method,
                   use_frontal,
                   batch_size,
                   data_path = '/media/samsung/chexpert/CheXpert-v1.0-small/',
                   use_upsampling=False,
                   upsampling_cols=['Cardiomegaly'],
                   smooth_lower=None,
                   smooth_upper=None):
    trainSet = CheXpertExtended(csv_path=data_path+'train.csv', 
                                 image_root_path=str(data_path),
                                 image_size=image_size,
                                 sample_frac=sample_frac,sample_seed=sample_seed,
                                 uncertainty_method=uncertainty_method,
                                 smooth_lower=smooth_lower,
                                 smooth_upper=smooth_upper,
                                 use_upsampling=use_upsampling,
                                 upsampling_cols=upsampling_cols,
                                 use_frontal=use_frontal,
                                shuffle=True,
                                verbose=False,
                                 mode='train', 
                                 class_index=-1)
    testSet =  CheXpertExtended(csv_path=data_path+'valid.csv',  
                                image_root_path=str(data_path), 
                                image_size=image_size,
                                uncertainty_method=None,
                                use_upsampling=False, 
                                use_frontal=use_frontal,
                                shuffle=False,
                                verbose=False,
                                mode='valid', 
                                class_index=-1)
    train_imratio = trainSet.imratio_list 
    trainloader =  torch.utils.data.DataLoader(trainSet, batch_size=batch_size, num_workers=12, shuffle=True)
    testloader =  torch.utils.data.DataLoader(testSet, batch_size=batch_size, num_workers=12, shuffle=False)
    
    return trainloader,testloader,train_imratio