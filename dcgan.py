import  torch
import torch.nn as nn
import os
import numpy as np

#Inspired from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html- But very much changed
class Generator(nn.Module):

    def create_generator_block(self, in_channels, last=False):
        '''
        Creating a generator dc-gan block. According to https://arxiv.org/abs/1511.06434
         consisting from a upsample of convTranspose, BatchNorm and Relu(if not the last layer)
        Upsample using stride of 2 , the out channels are divided by 2(/2 the input channels).
        :param in_channels:
        :param last:
        :return:
        '''

        if not last:
            out_channels=int(in_channels//2)
            layers_block_list = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                                    stride=2, padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True)]
        else:
            out_channels=3
            layers_block_list = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                                  stride=2, padding=1)]
        block = nn.Sequential(*layers_block_list)
        return block

    def __init__(self, noise_size=100, image_size=64):
        super(Generator, self).__init__()

        #The default structure of dcgan is built on 64 hidden size. starting from 4*4 image, there are 4 layers, each multiplies the size to by 2,
        #while the number of channels start from hidden size*16 and is divided each layer
        #If the image size is a different mutiple of 2 ,the hidden size is adjusted accrodingly since its a fully convolutinal cnn
        default_hidden_size = 64
        hidden_size_divider = (image_size // default_hidden_size)
        hidden_size = int(default_hidden_size // hidden_size_divider)


        #the number of channels is first hidden_size*16 and divided by two each layer, while the image size is upsampled.
        self.input_conv = nn.Sequential(nn.ConvTranspose2d(noise_size, hidden_size * 16, kernel_size=4, stride=1),
                                        nn.ReLU(inplace=True))
        self.convt2 = self.create_generator_block(in_channels=hidden_size * 16)
        self.convt3 = self.create_generator_block(in_channels=hidden_size * 8)
        self.convt4 = self.create_generator_block(in_channels=hidden_size * 4)
        self.convt5 = self.create_generator_block(in_channels=hidden_size * 2, last=True)
        self.output = nn.Tanh()


        #Initalization According to https://arxiv.org/abs/1511.06434
        for layer in self.modules():
            if isinstance(layer, (nn.ConvTranspose2d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
            if isinstance(layer, (nn.BatchNorm2d)):
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0)

    def forward(self, noise_inputs):

        input_conv_output = self.input_conv(noise_inputs)
        convt2_output = self.convt2(input_conv_output)
        convt3_output = self.convt3(convt2_output)
        convt4_output = self.convt4(convt3_output)
        convt5_output = self.convt5(convt4_output)
        output = self.output(convt5_output)

        return output

    def generate_noise(self, batch_size):
        #Generate noise from normal distribution
        noise = torch.randn(batch_size, self.noise_size, 1, 1)
        return noise

    def save_checkpoint(self,saved_dir, filename):
        # Save the weights to file
        torch.save(self.state_dict(), os.path.join(saved_dir, filename+".pth"))

    def load_from_checkpoint(self,saved_dir, filename):
        # Load the weights from file
        self.load_state_dict(torch.load(os.path.join(saved_dir, filename)))


class Discriminator(nn.Module):
    #Building DCGAN discrimnator . According to https://arxiv.org/abs/1511.06434
    #If as_critic=True , then the dicrimnator will become a critic ,outputting value instead of probability (https://arxiv.org/abs/1701.07875)


    def create_discriminator_block(self, in_channels, last=False, as_critic=False):
        #Discrimnator basic block. Consisting of Conv-BN-LeakyRelu. If built as critic , Normalization is changed to InstanceNorm-See https://arxiv.org/abs/1704.00028

        if not last:
            out_channels=2*in_channels
            padding=1
            layers_block_list = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=padding, bias=False)]
            if not as_critic:
                layers_block_list.append(nn.BatchNorm2d(out_channels))
            else:
                #If critic -BatchNorm is changed to InstanceNorm, becuase otherwise wasserstein gp -1 Lifshitz continues constaint won't work. See See https://arxiv.org/abs/1704.00028
                layers_block_list.append(nn.InstanceNorm2d(out_channels))
            layers_block_list.append(nn.LeakyReLU(negative_slope=0.2))
        else:
            out_channels=1
            padding=0
            layers_block_list = [nn.Conv2d(in_channels, out_channels, kernel_size=4,   stride=2, padding=padding)]

        block = nn.Sequential(*layers_block_list)
        return block


    def __init__(self, image_size=64, as_critic=False):
        super(Discriminator, self).__init__()

        #It is assumed image size is a multiplier power of 2 and is grater or equal to 64
        # The default structure of dcgan is built on 64 hidden size. There are 4 layers, each multiplies the channels by 2 and divided the spatial size by 2
        #Mirror of the Generator
        default_hidden_size=64
        hidden_size_divider=(image_size//default_hidden_size)
        hidden_size=int(default_hidden_size//hidden_size_divider)
        #Critic if wasserstein gan
        self.as_critic=as_critic
        self.input_conv = nn.Sequential(nn.Conv2d(3, hidden_size, kernel_size=4, stride=2, padding=1),
                                        nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = self.create_discriminator_block(hidden_size, as_critic=as_critic)
        self.conv3 = self.create_discriminator_block(hidden_size*2, as_critic=as_critic)
        self.conv4 = self.create_discriminator_block(hidden_size * 4, as_critic=as_critic)
        self.conv5=self.create_discriminator_block(hidden_size * 8,last=True, as_critic=as_critic)
        if not as_critic:
            self.sig_out = nn.Sigmoid()

        #Initalization According to https://arxiv.org/abs/1511.06434
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
            if isinstance(layer, (nn.BatchNorm2d) or nn.InstanceNorm2d):
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0)

    def forward(self, inputs):

        input_conv_output = self.input_conv(inputs)
        conv2_output = self.conv2(input_conv_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)

        #If critic, then the discrimnator outputs real number and not probability
        if not self.as_critic:
            output = self.sig_out(conv5_output)
        else:
            output=conv5_output
        return output

    def save_checkpoint(self,saved_dir, filename):
        # Save the weights to file
        torch.save(self.state_dict(), os.path.join(saved_dir, filename+".pth"))

    def load_from_checkpoint(self,saved_dir, filename):
        # Load the weights from file
        self.load_state_dict(torch.load(os.path.join(saved_dir, filename)))




