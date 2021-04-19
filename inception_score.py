from scipy.linalg import sqrtm
from torchvision.models import inception_v3
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spicy
from scipy import stats


INCEPTION_SIZE=299

def calc_inception_FID_score(batch_size, device, eval_dataset, generator, type):
    #calculate Frechete Inception score.
    #It is based on distnce of means and convarience of real imapes comapred to fake images

    #Inception model
    inception = inception_v3(pretrained=True)
    inception.fc = nn.Identity()
    inception.eval()
    inception= inception.to(device)


    data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=2,drop_last=True)

    all_real_images=None
    all_fake_images = None
    for ind, data in enumerate(data_loader):


        real_image = data[0].to(device)

        prepared_real_image = F.interpolate(real_image, INCEPTION_SIZE)
        inception_result_real = inception(prepared_real_image).cpu().detach().numpy()

        if not type=='SRGAN':
            noise = generator.module.generate_noise(real_image.size()[0]).to(device)
            fake_image = generator(noise).to(device).detach()
        else:
            lq_image=data[1]
            fake_image = generator(lq_image).to(device).detach()
        #Take Inception outputs- Before the classification layer
        prepared_fake_image = F.interpolate(fake_image, INCEPTION_SIZE)
        inception_result_fake = inception(prepared_fake_image).cpu().detach().numpy()

        if all_real_images is None:
            all_real_images=inception_result_real
            all_fake_images=inception_result_fake
        else:
            all_real_images=np.concatenate((all_real_images,inception_result_real))
            all_fake_images=np.concatenate((all_fake_images, inception_result_fake))

    #Calculate means
    mean_real_images = np.mean(all_real_images, axis=0)
    mean_fake_images = np.mean(all_fake_images, axis=0)

    # Calculate Covariance matrixes
    cov_real = np.cov(all_real_images, rowvar=False)
    cov_fake = np.cov(all_fake_images, rowvar=False)

    #Calculate the FID score
    means_distance=mean_real_images - mean_fake_images

    inception = means_distance.dot(means_distance) + np.trace(
        cov_real + cov_fake - 2 * sqrtm(cov_real.dot(cov_fake)))

    return inception


def calc_inception_score(device, generator, splits=10,eval_size=50000):
    #Calculate Inception score

    #Based on the classification output of the Inception model trained on Imagenet
    batch_size=32
    inception = inception_v3(pretrained=True)
    inception.eval()
    inception= inception.to(device)

    num_batches=eval_size//batch_size

    #claculate incpetion predictions in generated images
    all_fake_predictions = None
    for batch in range(num_batches):
        noise = generator.module.generate_noise(batch_size).to(device)
        fake_batch = generator(noise).to(device).detach()
        prepared_fake_batch = F.interpolate(fake_batch, INCEPTION_SIZE)
        inception_logits= inception(prepared_fake_batch)
        predictions=F.softmax(inception_logits,dim=-1).detach().cpu().numpy()
        if all_fake_predictions is None:
            all_fake_predictions=predictions
        else:
            all_fake_predictions=np.concatenate((all_fake_predictions,predictions))

    #Calculate Inception score . Based on KL Divergence.
    all_scores=[]
    for split in np.split(all_fake_predictions, splits):
        split_scores=[]
        prob_y = np.repeat(np.mean(split, axis=0, keepdims=True), len(split),axis=0)
        kl_div=stats.entropy(split, prob_y, axis=1)
        split_scores=np.exp(np.mean(kl_div))
        all_scores.append(split_scores)

    inception_score=np.mean(all_scores)


    return inception_score