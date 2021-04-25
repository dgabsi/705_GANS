import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import torch
import pandas as pd


'''
def create_celebA_dataset(data_dir, image_size, split='train'):

    transform =transforms.Compose([transforms.Resize((image_size ,image_size)),
                                   transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5) ,(0.5, 0.5, 0.5))
                                  ])

    celeba_dataset =datasets.CelebA(data_dir, split=split,transform=transform)

    return celeba_dataset
'''




class celebADataset(data.Dataset):

    def __init__(self, data_dir, image_size=64, split='train', type='generate'):

        super(celebADataset, self).__init__()

        split_list=['train', 'valid', 'test']
        #If dataset was not loaded - Loading it from pytorch dataset. this is the most comfortable way
        if len(os.listdir(os.path.join(data_dir, "celeba/img_align_celeba"))) == 0:
            print('Image directory is empty! Please read instructions in ReadMe file')

        self.images_dir = os.path.join(data_dir, "celeba/img_align_celeba")
        self.image_size=image_size
        #The split to train, valid, test
        self.split=split
        self.split_df=pd.read_csv(os.path.join(data_dir, "celeba/list_eval_partition.txt"), names=['file', 'split'], delim_whitespace=True)
        self.imgs=self.split_df.loc[self.split_df["split"]==split_list.index(self.split), ["file"]].values[:].squeeze()
        self.type=type

        #Collecting real images
        #for dirpath, _, files in os.walk(self.data_dir):
        #    for file in files:
        #        filename = os.path.join(dirpath, file)
        #        self.imgs.append(filename)
        #self.imgs=sorted(self.imgs)

        if type=='generate':    #Inspired from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html- But very much changed.
            self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])

        else:
            #This is for the super resolution Dataset strudture. Creating low reswolution and high resolution images
            self.lr_transform = transforms.Compose([transforms.Resize((image_size//4, image_size//4), Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
            self.transform = transforms.Compose([transforms.Resize((image_size, image_size), Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])




    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.images_dir, self.imgs[idx]))
        if self.type=="generate":
            return self.transform(image), torch.zeros(1)
        else:
            hr = self.transform(image)
            lr = self.lr_transform(image)
            return hr, lr

    def __len__(self):
        """
        return length of dataset. Api function for Dataset
        """
        return len(self.imgs)

