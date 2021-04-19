# 705 project
# Code for course Deep Learning for Image Analysis 
### Daniela Stern- Gabsi 

### github- dgabsi/SLNI_Project706
(updates were made from danielaneuralx which is my working github but its all mine)


This project explores a few Generative Adversarial Networks(GAN) techniques .
This project is based on the celebA dataset, which is a large scale dataset of human faces.
I will explore DCGAN which was the first convolutional GAN, the Wassertstein GAN with gradient penalty,
and the Self Attention GAN (SAGAN) which is based on the attention mechanism.
I will measure the performance using the Frechet Inception distance (FID)

I have also used TensorBoard for following the training.


##Very Important: Due to the size of data, I have put the data files in OneDrive.
In One Drive please take contents of For706coursework and put under the directory saved_models
https://cityuni-my.sharepoint.com/personal/daniela_stern-gabsi_city_ac_uk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fdaniela%5Fstern%2Dgabsi%5Fcity%5Fac%5Fuk%2FDocuments%2FFor706coursework


Main notebook:
**GAN_main_notebook.ipynb** 

#Please make sure you have the following structure 
Project structure:
- celeba_data (Directory for data)
  - celeba (The data will be downloaded here)
- experiments (Directory for Tensorboard logs)
- saved(Directory for saved models, pickles and visualization results charts- **models from drive should be put directly here**)
- dcgan.py 
- sagan.py
- main_gan.py
- training_gan.py
- utils.py
- inception_score.py  (Dataset)
- celebA_dataset.py (Dataset)

packages needed :
- torch 1.8.0 (I used with cu111)
- datetime
- time
- transformers 4.4.1
- matplotlib 3.3.4
- numpy 1.20.1
- pandas 1.2.3
- scikit-learn 0.24.1
- tensorboard 2.4.1
- torchtext 0.9.0
- spacy 3.0.5
- pickle
- bertviz 1.0.0
