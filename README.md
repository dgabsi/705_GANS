# 705 project
# Code for course Deep Learning for Image Analysis 
### Daniela Stern- Gabsi 

### github- dgabsi/705_GANS
(updates were made from danielaneuralx which is my working github but its all mine)


This project explores a few Generative Adversarial Networks(GAN) techniques .
This project is based on the celebA dataset, which is a large scale dataset of human faces.
I will explore DCGAN which was the first convolutional GAN, the Wassertstein GAN with gradient penalty,
and the Self Attention GAN (SAGAN) which is based on the attention mechanism.
In addition, I will implement the SRGAN for improve image resolution.
I will measure the performance using the Frechet Inception distance (FID)

I have also used TensorBoard for following the training.


##Very Important: Due to the size of data, I have put the data files in OneDrive.
In One Drive celeba directory please take the file  img_align_celeba.zip and put under the directory :
celeba_data-> celeba and unzip it. this will create directory img_align_celeba under celeba_data-> celeba
https://cityuni-my.sharepoint.com/personal/daniela_stern-gabsi_city_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdaniela%5Fstern%2Dgabsi%5Fcity%5Fac%5Fuk%2FDocuments%2F705data%5Fceleba%2Fceleba

Main notebook:
**GAN_main_notebook.ipynb** 

#Please make sure you have the following structure 
Project structure:
- celeba_data (Directory for data)
  - celeba (In this directory you have to put the zip file from MyDrive and unzip it)
    - img_align_celeba (This directory includes all images. It will be created after you unzip img_align_celeba.zip
    - list_eval_partition.txt
- experiments (Directory for Tensorboard logs)
- saved(Directory for saved models, pickles and visualization results charts)
- images (directory for showing images in jupyter)  
- dcgan.py 
- sagan.py
- main_gan.py
- training_gan.py
- utils.py
- inception_score.py  
- celebA_dataset.py (Dataset)
- **GAN_main_notebook.ipynb** (Main notebook that should be used)

packages needed :
- torch 1.8.1 
- torchvision 0.9.1
- datetime
- time
- matplotlib 3.3.4
- numpy 1.20.1
- pandas 1.2.3
- scikit-learn 0.24.1
- tensorboard 2.4.1
- pickle
- pillow 8.2.0
- scipy
