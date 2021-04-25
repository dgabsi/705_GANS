import math

import  torch
import torch.nn as nn
from utils import show_images_grid
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from inception_score import calc_inception_score, calc_inception_FID_score
from utils import plot_all_losses, plot_all_accuracies, plot_inception_scores
from torch.nn.functional import mse_loss, relu
from torchvision.models import vgg19
import utils


def calc_losses_dcgan(batch_real, generator, discriminator, optim_generator, optim_discr, device, noise_size=100):
    #Calculate loss of DCAN
    #The loss is base the adversial loss of the dicrimnator and the generator.
    #Based on binary cross entropy loss.


    criterion = nn.BCELoss()

    #The dicrimnator loss is made of discrimnator on fake images +loss of generated images(fake)
    optim_discr.zero_grad()
    noise_for_discr = utils.generate_noise(batch_real.size()[0], noise_size).to(device)
    fake_for_discr = generator(noise_for_discr).to(device)
    disc_fake_pred = discriminator(fake_for_discr.detach()).view(-1)

    noise_fake_discr = torch.randn_like(disc_fake_pred) * 0.1
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred)+noise_fake_discr)
    disc_real_pred = discriminator(batch_real).view(-1)
    # Adding some noise for better convergence
    noise_real_disc = torch.randn_like(disc_real_pred) * 0.1
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred)-noise_real_disc)

    # total loss is the loss on fake + loss on real. multiplied by 0.5 each to get max of 1.
    total_discr_loss = 0.5* disc_fake_loss +0.5*disc_real_loss
    total_discr_loss.backward()
    optim_discr.step()

    ##Generator loss
    optim_generator.zero_grad()
    noise_gener = utils.generate_noise(batch_real.size()[0], noise_size).to(device)
    fake = generator(noise_gener).to(device)
    discr_fake_pred_for_gener = discriminator(fake).view(-1)

    #Adding some noise for better convergence
    noise_gener = torch.randn_like(discr_fake_pred_for_gener) * 0.1
    gener_loss = criterion(discr_fake_pred_for_gener, torch.ones_like(discr_fake_pred_for_gener)-noise_gener)
    gener_loss.backward()
    optim_generator.step()

    #Calculate accruacies of the discriminator
    batch_correct_discr_real = (disc_real_pred.detach().cpu()> 0.5).float().numpy().sum()
    batch_correct_discr_fake = (disc_fake_pred.detach().cpu()<= 0.5).float().numpy().sum()

    batch_accuracy_real = batch_correct_discr_real * 100 / batch_real.size(0)
    batch_accuracy_fake = batch_correct_discr_fake * 100 / batch_real.size(0)
    return gener_loss.item(), total_discr_loss.item(), batch_accuracy_real, batch_accuracy_fake


def calc_losses_wsgan(batch_real, generator, critic, optim_generator, optim_critic, device,num_iter_critic=5, lambda_penalty=10, noise_size=64):
    # Calculate loss of Wasserstein GAN with Gradient Penalty. See https://arxiv.org/abs/1701.07875, https://arxiv.org/abs/1704.00028
    #The dicrimnator is now a critic and the loss is caculated based Earth movers loss +gradient penalty

    criterion = nn.MSELoss()
    critic_loss_avg = 0

    batch_size=batch_real.size()[0]

    #A few iteration of the discrimator for each generator loss
    for iter in (range(num_iter_critic)):
        optim_critic.zero_grad()
        # Generating fake image
        noise_for_critic = utils.generate_noise(batch_size, noise_size).to(device)
        fake_for_critic = generator(noise_for_critic).to(device)

        # critic on fake
        critic_fake_pred = critic(fake_for_critic.detach()).view(-1)
        # critic on real
        critic_real_pred = critic(batch_real.detach()).view(-1)

        # Gradient penalty-Enforcing 1 Lifshitz continuous https://arxiv.org/abs/1704.00028
        channels, height,  width = batch_real.size()[1:]
        epsilon_mix = torch.rand(batch_size,1,1,1, requires_grad=True).repeat(1, channels, height,  width).to(device)
        #epsilon_mix.requiers_grad=True
        mixed_images = batch_real * epsilon_mix + fake_for_critic * (1. - epsilon_mix)
        critic_pred_mixed = critic(mixed_images)
        grad_mixed = torch.autograd.grad(inputs=mixed_images,
                                         outputs=critic_pred_mixed,
                                         grad_outputs=torch.ones_like(critic_pred_mixed, requires_grad=False),
                                         create_graph=True,
                                         retain_graph=True
                                         )[0]
        norm_grad_mixed = grad_mixed.flatten(1).norm(2, dim=1)
        grad_penalty = lambda_penalty * torch.mean((norm_grad_mixed-1.)**2)

        #This is the wasserrstein loss-Earth Movers loss + gradient penalty for enforcing 1 Lifshitz continuous
        critic_loss = -(critic_real_pred.mean()-critic_fake_pred.mean() ) + grad_penalty

        critic_loss_avg += critic_loss.item()
        critic_loss.backward(retain_graph=True)
        optim_critic.step()

    critic_loss_avg = critic_loss_avg / num_iter_critic

    # Generator loss
    optim_generator.zero_grad()
    noise_gener = utils.generate_noise(batch_size, noise_size).to(device)
    fake = generator(noise_gener).to(device)
    critic_fake_pred_for_gener = critic(fake).view(-1)
    #We minimise the critic mean value  on the generator weights
    #multiply by -1 for gradient ascent
    gener_loss = -critic_fake_pred_for_gener.mean()
    gener_loss.backward()
    optim_generator.step()
    return gener_loss.item(), critic_loss_avg*(-1)

def calc_losses_sagan(batch_real, generator, discriminator, optim_generator, optim_discr, device, noise_size):

    # Calculate loss of according to https://arxiv.org/abs/1805.08318
    #Hinge loss for the dicrimnator
    discriminator.train()
    generator.train()
    optim_discr.zero_grad()
    noise_for_discr = utils.generate_noise(batch_real.size()[0], noise_size).to(device)
    fake_for_discr = generator(noise_for_discr).to(device)

    discr_fake_pred = discriminator(fake_for_discr)

    #Dicrimnator loss is the Hinge loss of real image and fake images
    disc_fake_loss = relu(1.0 + discr_fake_pred).mean()
    disc_real_pred = discriminator(batch_real)
    disc_real_loss = relu(1.0 - disc_real_pred).mean()

    # total loss is the loss on fake + loss on real
    total_discr_loss = disc_fake_loss + disc_real_loss
    total_discr_loss.backward()
    optim_discr.step()

    # Generator loss
    generator.train()
    #discriminator.eval()
    optim_generator.zero_grad()
    noise_gener = utils.generate_noise(batch_real.size()[0], noise_size).to(device)
    fake = generator(noise_gener).to(device)
    discr_fake_pred_for_gener = discriminator(fake)
    # We minimise the dicrimnator mean value  on the generator weights
    # multiply by -1 for gradient ascent
    gener_loss = -discr_fake_pred_for_gener.mean()
    # gener_loss = criterion(discr_fake_pred_for_gener, torch.ones_like(discr_fake_pred_for_gener))
    gener_loss.backward()
    optim_generator.step()

    # Calculate accuracies of the discriminator
    batch_correct_discr_real = (disc_real_pred.detach().cpu() > 0).float().numpy().sum()
    batch_correct_discr_fake = (discr_fake_pred.detach().cpu() < 0).float().numpy().sum()

    batch_accuracy_real = batch_correct_discr_real * 100 / batch_real.size(0)
    batch_accuracy_fake = batch_correct_discr_fake * 100 / batch_real.size(0)

    return gener_loss.item(),  total_discr_loss.item(), batch_accuracy_real, batch_accuracy_fake


def calc_losses_srgan(batch_real_hq, batch_lq, generator, discriminator, optim_generator, optim_discr, vgg_output, device):
    ##The SRGAN according to https://arxiv.org/abs/1609.04802, https://arxiv.org/pdf/1903.09922.pdf
    vgg_output=vgg_output.to(device)
    batch_lq=batch_lq.to(device)

    #Calculate the vgg loss based on L1 loss- see https://arxiv.org/pdf/1903.09922.pdf
    criterion_l1 = nn.L1Loss()
    #Bce loss for Adversial loss
    criterion_bce=nn.BCEWithLogitsLoss()
    criterion_mse = nn.MSELoss()

    optim_discr.zero_grad()


    #Discriminator loss
    batch_hq_fake=generator(batch_lq.detach()).to(device)
    discr_fake_hq_pred = discriminator(batch_hq_fake.detach())
    discr_real_hq_pred = discriminator(batch_real_hq)

    #Some noise for better convergence
    noise_fake_discr = torch.rand_like(discr_fake_hq_pred).to(device) * 0.1
    noise_real_discr = torch.rand_like(discr_real_hq_pred).to(device)* 0.1
    disc_fake_hq_loss=0.5*criterion_bce(discr_fake_hq_pred, torch.zeros_like(discr_real_hq_pred, requires_grad=False)+noise_fake_discr)
    disc_real_hq_loss = 0.5 * criterion_bce(discr_real_hq_pred, torch.ones_like(discr_real_hq_pred, requires_grad=False)-noise_real_discr)
    total_discr_loss = disc_fake_hq_loss + disc_real_hq_loss
    total_discr_loss.backward()
    optim_discr.step()

    #Generator loss
    optim_generator.zero_grad()
    generator_fake_pred = discriminator(batch_hq_fake)
    gener_discr_loss=1e-3*criterion_bce(generator_fake_pred, torch.ones_like(discr_fake_hq_pred, requires_grad=False))
    vgg_outputs_real=vgg_output(batch_real_hq.detach())
    vgg_outputs_fake=vgg_output(batch_hq_fake)
    gener_content_loss=criterion_l1(vgg_outputs_real,vgg_outputs_fake)+criterion_mse(batch_real_hq,batch_hq_fake)

    gener_loss = gener_discr_loss+gener_content_loss
    gener_loss.backward()
    optim_generator.step()

    return gener_loss.item(), total_discr_loss.item()



def train_gan(num_epochs, batch_size, noise_size, device, train_dataset, val_dataset, generator, discriminator,  config, type='DCGAN',run_dir=None, saved_dir=None, run_name='', verbose_show=50, calc_IS=True):
    num_images_tensorboard = 24
    patience=0
    NUM_PATIENCE=5
    prev_FID_score=-math.inf

    vgg = vgg19(pretrained=True).eval()
    for param in vgg.parameters():
        param.requires_grad=False
    vgg= nn.Sequential(*list(vgg.features[:18])).to(device)



    data_loader = DataLoader(train_dataset, batch_size=batch_size ,shuffle=True)

    # print("#finish dl")


    generator_optimizer = optim.Adam(generator.parameters(), **config['optim_gen'])
    discriminator_optimizer = optim.Adam(discriminator.parameters(), **config['optim_disc'])
    global_step =0

    writer = SummaryWriter(os.path.join(run_dir, run_name))

    #Prepare constnat noise from tensorboard or constant images for SRGAN
    if type=='SRGAN':
        data=next(iter(data_loader))
        tensorboard_real=data[0]
        tensorboard_fake=data[1].to(device)
    else:
        noise_for_tesorboard = utils.generate_noise(num_images_tensorboard,noise_size=noise_size).to(device)
    generator.train()
    discriminator.train()

    all_gener_losses =[]
    all_discr_losses =[]
    all_real_accuracy=[]
    all_fake_accuracy = []
    inception_FID_scores=[]
    inception_scores = []
    for epoch in range(num_epochs):
        for num_batch, data in enumerate(data_loader):

            batch_real =data[0]
            batch_real =batch_real.to(device)
            # Training Discriminator

            #Different loss for each GAN. Sending each batch to the specific loss according to the type of GAN
            if type=='DCGAN':
                gener_loss,  discr_loss, batch_accuracy_real, batch_accuracy_fake=calc_losses_dcgan(batch_real, generator, discriminator, generator_optimizer, discriminator_optimizer, device,noise_size=noise_size)
                if not num_batch % verbose_show:
                    print (f"Batch:{num_batch}/Epoch: {epoch} Discriminator loss: {discr_loss:.3f} Generator loss: {gener_loss:.3f} , Acc real:{batch_accuracy_real:.3f} Acc fake:{batch_accuracy_fake:.3f}")
                all_real_accuracy.append(batch_accuracy_real)
                all_fake_accuracy.append(batch_accuracy_fake)

            if type=='WSGAN':
                gener_loss,  discr_loss=calc_losses_wsgan(batch_real, generator, discriminator, generator_optimizer, discriminator_optimizer, device, num_iter_critic=config['iter_critic'], lambda_penalty=config['lambda_penalty'],noise_size=noise_size)
                if not num_batch % verbose_show:
                    print(f"Batch{num_batch}/Epoch: {epoch} Critic: {discr_loss:.3f} Generator loss: {gener_loss:.3f} ")

            if type=='SAGAN':
                gener_loss, discr_loss, _, _ = calc_losses_sagan(batch_real,  generator,  discriminator,  generator_optimizer,  discriminator_optimizer, device,noise_size=noise_size)
                if not num_batch % verbose_show:
                    print(
                    f"Batch{num_batch}/Epoch {epoch} Discriminator: {discr_loss:.3f} Generator loss: {gener_loss:.3f}")
            if type=='SRGAN':
                gener_loss, discr_loss= calc_losses_srgan(batch_real , data[1],generator, discriminator,generator_optimizer, discriminator_optimizer, vgg,device)
                if not num_batch % verbose_show:
                    print(
                        f"Batch{num_batch}/Epoch {epoch} Discriminator: {discr_loss:.3f} Generator loss: {gener_loss:.3f} ")



            all_gener_losses.append(gener_loss)
            all_discr_losses.append(discr_loss)

            #Outputs for Tensorboard
            with torch.no_grad():
                if not num_batch % 1000:
                    if type=='SRGAN':
                        fake_images = generator(tensorboard_fake).to(device)
                        batch_real=tensorboard_real
                    else:
                        fake_images = generator(noise_for_tesorboard).to(device)
                    fake_images_display=fake_images[:num_images_tensorboard].detach().cpu()
                    real_images_display = batch_real[:num_images_tensorboard].detach().cpu()
                    img_grid_fake = show_images_grid(fake_images_display, title=f"{type} Epoch: {epoch}")
                    img_grid_real =show_images_grid(real_images_display.cpu(), title="Real images")
                    writer.add_image("Real", img_grid_real, global_step=global_step)
                    writer.add_image("Fake", img_grid_fake, global_step=global_step)

                writer.add_scalar("Generator loss", gener_loss, global_step=global_step)
                if ((type=='DCGAN') or (type=='SAGAN')):
                    writer.add_scalar("Discriminator loss", discr_loss, global_step=global_step)
                else:
                    writer.add_scalar("Critic loss", discr_loss, global_step=global_step)
                    global_step += 1

        #Calculation of scores on the evaluation dataset
        inception_FID_score=calc_inception_FID_score(batch_size, device, val_dataset, generator, type,noise_size)
        inception_FID_scores.append(inception_FID_score)
        if calc_IS:
            inception_score = calc_inception_score(device=device, noise_size=noise_size, generator=generator, eval_size=len(val_dataset))
            inception_scores.append(inception_score)
            print(f"Epoch:{epoch} FID score:{inception_FID_score:.3f}  Inception score:{inception_score:.3f} ")
        else:
            print(f"Epoch:{epoch} FID score:{inception_FID_score:.3f}")

        #At the end of epoch save weights to checkpoint
        discr_filename=f"{run_name}_Discriminator_epoch_{epoch}"
        utils.save_checkpoint(discriminator, saved_dir, discr_filename)
        gener_filename = f"{run_name}_Generator_epoch_{epoch}"
        utils.save_checkpoint(generator,saved_dir, gener_filename)


        if prev_FID_score < inception_FID_score:
            patience += 1
            if patience >=NUM_PATIENCE:
                print(f"FID score has increased for at least {NUM_PATIENCE} epochs . Stopping training.")
                break
        else:
            patience=0
        prev_FID_score=inception_FID_score

    plot_all_losses(all_gener_losses, all_discr_losses, f"{run_name} GAN_losses", saved_dir, f"{run_name}_gan_losses",'batches')
    #if ((type=='DCGAN') or (type=='SAGAN')):
    #    plot_all_accuracies(all_fake_accuracy, all_real_accuracy, f"{type} GAN_accuracy", saved_dir, f"{type}_gan_accuracy",'batches')

    plot_inception_scores([inception_FID_scores],[f"{run_name}_FID"], f"{type} GAN_FID", saved_dir, f"{run_name}_gan_FID", "FID")
    if calc_IS:
        plot_inception_scores([inception_scores], [f"{run_name}_IS"], f"{type} GAN_Inception_score", saved_dir, f"{run_name}_gan_IS", "IS")

    #If it is SRGAN we will not calculate FID score
    if calc_IS:
        return inception_FID_scores, inception_scores
    else:
        return inception_FID_scores, None

