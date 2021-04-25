import numpy as np
import random
import torch.utils.data
import torch.nn as nn

import utils
from celebAdataset import celebADataset #create_celebA_dataset
from dcgan import Generator, Discriminator
from training_gan import train_gan
from torchsummary import summary
from inception_score import calc_inception_score, calc_inception_FID_score
from sagan import SAGenerator, SADiscriminator
from srgan import SRGenerator, SRDiscriminator
import os
from utils import save_to_pickle
import datetime

DATA_DIR='./celeba_data'
IMAGE_SIZE=64 #Should be a multiplication of 2
RUNS_DIR='./experiments'
MODELS_DIR='./saved'
BATCH_SIZE=64



NOISE_SIZE_DCGAN=100
OPTIM_DCGAN_GENERATOR={'lr':0.0002, 'betas': (0.5, 0.999)}
OPTIM_DCGAN_DISCRIMINATOR={'lr':0.0002, 'betas': (0.5, 0.999)}
OPTIM_DCGAN={'optim_gen':OPTIM_DCGAN_GENERATOR, 'optim_disc':OPTIM_DCGAN_DISCRIMINATOR}


OPTIM_WSGAN_GENERATOR={'lr':2e-4, 'betas': (0.0, 0.9)}
OPTIM_WSGAN_DISCRIMINATOR={'lr':2e-4, 'betas': (0.0, 0.9)}
OPTIM_WSGAN={'optim_gen':OPTIM_WSGAN_GENERATOR, 'optim_disc':OPTIM_WSGAN_DISCRIMINATOR}
LAMBDA_PENALTY = 10
NOISE_SIZE_WS_GAN=100

NOISE_SIZE_SA_GAN=100
OPTIM_SAGAN_GENERATOR={'lr':1e-4, 'betas': (0.0, 0.9)}
OPTIM_SAGAN_DISCRIMINATOR={'lr':4e-5, 'betas': (0.0, 0.9)}
OPTIM_SAGAN={'optim_gen':OPTIM_SAGAN_GENERATOR, 'optim_disc':OPTIM_SAGAN_DISCRIMINATOR}


OPTIM_SRGAN_GENERATOR={'lr':1e-4, 'betas': (0.5, 0.999)}
OPTIM_SRGAN_DISCRIMINATOR={'lr':1e-5, 'betas': (0.5, 0.999)}
OPTIM_SRGAN={'optim_gen':OPTIM_SRGAN_GENERATOR, 'optim_disc':OPTIM_SRGAN_DISCRIMINATOR}
SR_IMAGE_SIZE=96


def run_dcgan(device, image_size, noise_size, batch_size, config, run_dir, saved_dir, run_name, num_epochs, val_dataset, train_dataset=None, checkpoints=None, mode='train', gpu_num=1):

    #Run DCGAN
    type = 'DCGAN'
    dcgan_generator = Generator(noise_size=noise_size, image_size=image_size).to(device)
    dcgan_discriminator = Discriminator(image_size=image_size).to(device)
    #Parallel for improved performence
    if device.type=='cuda' and gpu_num>1:
        dcgan_generator=nn.DataParallel(dcgan_generator,list(range(gpu_num)))
        dcgan_discriminator = nn.DataParallel(dcgan_discriminator, list(range(gpu_num)))

    #Print networks
    print('Discriminator')
    summary(dcgan_discriminator, (3, image_size, image_size))
    print('Generator')
    summary(dcgan_generator, (noise_size,1,1))

    if checkpoints is not None:
        utils.load_from_checkpoint(dcgan_generator,saved_dir, checkpoints["generator"])
        utils.load_from_checkpoint(dcgan_discriminator, saved_dir, checkpoints["discriminator"])

    run_name='DCGAN'+'_'+run_name
    #We train the model in train phase and only calculate scores in test mode
    if mode=='train':
        inception_FID_scores, inception_scores=train_gan(num_epochs, batch_size, noise_size, device, train_dataset, val_dataset, dcgan_generator, dcgan_discriminator,
                    type='DCGAN', config=config, run_dir=run_dir, saved_dir=saved_dir, run_name=run_name)
    elif mode=='test':
       inception_FID_scores = [calc_inception_FID_score(batch_size, device, val_dataset, dcgan_generator,type,noise_size)]
       inception_scores = [calc_inception_score(device, noise_size, dcgan_generator,eval_size=len(val_dataset))]
    #Return list of all score accumulated in epochs



    date_str = datetime.datetime.now().strftime("%m%d%Y%H")
    save_to_pickle(inception_FID_scores, os.path.join(saved_dir, 'dcgan_fid_'+run_name+date_str+".pickle") )
    save_to_pickle(inception_scores, os.path.join(saved_dir, 'dcgan_IS_'+ run_name+date_str + ".pickle"))

    return inception_FID_scores, inception_scores



def run_wsgan(device, image_size, noise_size, batch_size, config, run_dir, saved_dir, run_name, num_epochs, val_dataset, train_dataset=None, checkpoints=None, mode='train', gpu_num=1):
    type = 'WSGAN'

    wsgan_generator = Generator(noise_size=noise_size, image_size=image_size).to(device)
    wsgan_critic = Discriminator(image_size=image_size, as_critic=True).to(device)
    # Parallel for improved performence
    if ((device.type == 'cuda') and (gpu_num>1)):
        wsgan_generator = nn.DataParallel(wsgan_generator, list(range(gpu_num)))
        wsgan_critic = nn.DataParallel(wsgan_critic, list(range(gpu_num)))
    # Print networks
    print('Critic')
    summary(wsgan_critic, (3, image_size, image_size))
    print('Generator')
    summary(wsgan_generator, (noise_size, 1, 1))

    if checkpoints is not None:
        utils.load_from_checkpoint(wsgan_generator, saved_dir, checkpoints["generator"])
        utils.load_from_checkpoint(wsgan_critic, saved_dir, checkpoints["discriminator"])

    run_name='WSGAN'+'_'+run_name
    if mode=='train':
        inception_FID_scores, inception_scores=train_gan(num_epochs, batch_size, noise_size, device, train_dataset, val_dataset, wsgan_generator, wsgan_critic,
                    type='WSGAN', config=config, run_dir=run_dir, saved_dir=saved_dir, run_name=run_name)
    elif mode=='test':
        inception_FID_scores = [calc_inception_FID_score(batch_size, device, val_dataset, wsgan_generator,type,noise_size)]
        inception_scores = [calc_inception_score(device, noise_size,wsgan_generator,eval_size=len(val_dataset))]

    date_str = datetime.datetime.now().strftime("%m%d%Y%H")
    save_to_pickle(inception_FID_scores, os.path.join(saved_dir, 'wsgan_fid_'+run_name+ date_str + ".pickle"))
    save_to_pickle(inception_scores, os.path.join(saved_dir, 'wsgan_IS_' +run_name+date_str + ".pickle"))

    return inception_FID_scores, inception_scores


def run_sagan(device, image_size, noise_size, batch_size, config, run_dir, saved_dir, run_name, num_epochs, val_dataset, train_dataset=None, checkpoints=None, mode='train', gpu_num=1):
    type = 'SAGAN'
    sagan_generator = SAGenerator(noise_size=noise_size, image_size=image_size).to(device)
    sagan_discriminator = SADiscriminator(image_size=image_size).to(device)
    # Parallel for improved performance
    if ((device.type == 'cuda') and (gpu_num>1)):
        sagan_generator = nn.DataParallel(sagan_generator, list(range(gpu_num)))
        sagan_discriminator = nn.DataParallel(sagan_discriminator, list(range(gpu_num)))
    # Print networks
    print('Discriminator')
    summary(sagan_discriminator, (3, image_size, image_size))
    print('Generator')
    summary(sagan_generator, (noise_size, 1, 1))
    if checkpoints is not None:
        utils.load_from_checkpoint(sagan_generator, saved_dir, checkpoints["generator"])
        utils.load_from_checkpoint(sagan_discriminator, saved_dir, checkpoints["discriminator"])

    run_name='SAGAN'+'_'+run_name
    if mode=='train':
        inception_FID_scores, inception_scores=train_gan(num_epochs, batch_size, noise_size, device, train_dataset, val_dataset, sagan_generator, sagan_discriminator,
                    type='SAGAN', config=config, run_dir=run_dir, saved_dir=saved_dir, run_name=run_name)
    elif mode=='test':
        inception_FID_scores = [calc_inception_FID_score(batch_size, device, val_dataset, sagan_generator,type,noise_size)]
        inception_scores = [calc_inception_score(device,noise_size, sagan_generator,eval_size=len(val_dataset))]

    date_str = datetime.datetime.now().strftime("%m%d%Y%H")
    utils.save_to_pickle(inception_FID_scores, os.path.join(saved_dir, 'sagan_fid_' +run_name+ date_str + ".pickle"))
    utils.save_to_pickle(inception_scores, os.path.join(saved_dir, 'sagan_IS_' + run_name+date_str + ".pickle"))

    return inception_FID_scores, inception_scores


def run_srgan(device, image_size, batch_size, config, run_dir, saved_dir, run_name, num_epochs, val_dataset, train_dataset=None, checkpoints=None, mode='train', gpu_num=1):

    srgan_generator = SRGenerator(device).to(device)
    srgan_discriminator = SRDiscriminator().to(device)
    summary(srgan_discriminator, (3, image_size, image_size))
    summary(srgan_generator, (3, image_size//4, image_size//4))

    if checkpoints is not None:
        utils.load_from_checkpoint(srgan_generator, saved_dir, checkpoints["generator"])
        utils.load_from_checkpoint(srgan_discriminator, saved_dir, checkpoints["discriminator"])

    run_name='SRGAN'+'_'+run_name
    if mode=='train':
        inception_FID_scores=train_gan(num_epochs, batch_size, None,device, train_dataset, val_dataset, srgan_generator, srgan_discriminator,
                    type='SRGAN', config=config, run_dir=run_dir, saved_dir=saved_dir, run_name=run_name,calc_IS=False)
    elif mode=='test':
        inception_FID_scores=[calc_inception_FID_score(batch_size, device, val_dataset, srgan_generator,type='SRGAN')]

    date_str = datetime.datetime.now().strftime("%m%d%Y%H")
    utils.save_to_pickle(inception_FID_scores, os.path.join(saved_dir, 'srgan_fid_' + date_str + ".pickle"))

    return inception_FID_scores



if __name__ == '__main__':
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(device)

    OPTIM_WSGAN = {'optim_gen': OPTIM_WSGAN_GENERATOR, 'optim_disc': OPTIM_WSGAN_DISCRIMINATOR, 'iter_critic': 5,
                   'lambda_penalty': LAMBDA_PENALTY}

    #rain_celebA=celebADataset('./celeba_data/', image_size=IMAGE_SIZE, split='train', type="sr")
    train_celebA = celebADataset('./celeba_data/', image_size=IMAGE_SIZE, split='train')
    print(f"Training images: {len(train_celebA)}")
    val_celebA = celebADataset('./celeba_data/', image_size=IMAGE_SIZE, split='valid') #celebA_dataset(root=DATA_DIR, split='valid')
    print(f"Validation images: {len(val_celebA)}")

    #inception_FID_scores, inception_scores=run_dcgan(device, IMAGE_SIZE, NOISE_SIZE_DCGAN, BATCH_SIZE, OPTIM_DCGAN, RUNS_DIR, MODELS_DIR,
    #         'dcgan', 3, val_celebA , train_dataset=train_celebA, checkpoints=None, mode='train')

    run_wsgan(device, IMAGE_SIZE, NOISE_SIZE_WS_GAN, BATCH_SIZE, OPTIM_WSGAN, RUNS_DIR, MODELS_DIR,
            'wsgan-test', 3, val_celebA, train_dataset=train_celebA, checkpoints=None, mode='train', gpu_num=2)

    inception_FID_scores, inception_scores = run_sagan(device, IMAGE_SIZE, NOISE_SIZE_SA_GAN, BATCH_SIZE,
                                                       OPTIM_SAGAN, RUNS_DIR, MODELS_DIR,
                                                       'sagan', num_epochs=15, val_dataset=val_celebA,
                                                       train_dataset=train_celebA, checkpoints=None, mode='train',
                                                       gpu_num=1)

    OPTIM_SRGAN_GENERATOR = {'lr': 1e-4, 'betas': (0.9, 0.999)}
    OPTIM_SRGAN_DISCRIMINATOR = {'lr': 4e-4, 'betas': (0.9, 0.999)}
    OPTIM_SRGAN = {'optim_gen': OPTIM_SRGAN_GENERATOR, 'optim_disc': OPTIM_SRGAN_DISCRIMINATOR}
    SR_IMAGE_SIZE = 64
    SR_GAN_BATCH_SIZE = 64
    train_sr_celebA = celebADataset('./celeba_data/', image_size=SR_IMAGE_SIZE, split='train', type='srgan')
    print(f"Training images: {len(train_sr_celebA )}")
    val_sr_celebA = celebADataset('./celeba_data/', image_size=SR_IMAGE_SIZE, split='valid',
                                  type='srgan')  # celebA_dataset(root=DATA_DIR, split='valid')
    print(f"Validation images: {len(val_sr_celebA )}")


    inception_FID_scores = run_srgan(device, SR_IMAGE_SIZE, SR_GAN_BATCH_SIZE, OPTIM_SRGAN, RUNS_DIR, MODELS_DIR,
                                     'srgan', 12, val_sr_celebA, train_dataset=train_sr_celebA, checkpoints=None,
                                     mode='train', gpu_num=1)

    #run_wsgan(device, IMAGE_SIZE, noise_size, feature_mul_factor, batch_size, config_optim, run_dir, saved_dir,
    #          run_name, num_epochs, val_dataset, train_dataset=None, checkpoints=None, mode='train'):

    #device, image_size, batch_size, config_optim, run_dir, saved_dir, run_name, num_epochs, val_dataset, train_dataset=None, checkpoints=None, mode='train'