import torch
import torch.nn as nn
import os


class SRGenerator(nn.Module):
    '''
        Generator for SRGAN
        Built according to https://arxiv.org/abs/1609.04802
        With adjustments according to https://arxiv.org/pdf/1903.09922.pdf
    '''

    class SRGeneratorResBlock(nn.Module):
        # Sequence of Generator basic blocks -Each block (conv-bn-prelu)+Conv+bn with a residual connection. Number of channels is constant.
        def __init__(self, device, channels, num_blocks=16):

            super().__init__()

            self.res_blocks=[]
            for block in range(num_blocks):
                layers_block_list = [nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                             nn.BatchNorm2d(channels)]
                layers_block_list.append(nn.PReLU())
                layers_block_list.append(nn.Conv2d(channels, channels, kernel_size=3,padding=1))
                layers_block_list.append(nn.BatchNorm2d(channels))
                self.res_blocks.append(nn.Sequential(*layers_block_list).to(device))

        def forward(self, x):
            for layer in self.res_blocks:
                #Residual conn
                x=layer(x)+x
            return x

    def create_pixel_shuffle_block(self,channels, num_blocks=2):
        # Sequence of Generator pixel shift block -Each block (conv-pixelshuffle-prelu)+Conv+bn with a residual connection. Number of channels is constant.
        #This will upsample the image by 4 (2 blocks and each upsample by 2) and also the channels
        pixel_shuffle_b_list=[]
        for block in range(num_blocks):
            pixel_shuffle_b_list.append(nn.Sequential(nn.Conv2d(channels, 4*channels, kernel_size=3, padding=1),
                      nn.PixelShuffle(upscale_factor=2),
                      nn.PReLU()))

        return nn.Sequential(*pixel_shuffle_b_list)



    def __init__(self, device, gan_basis=64):
        super(SRGenerator, self).__init__()

        self.layer1=nn.Sequential(nn.Conv2d(3, gan_basis, kernel_size=9, padding=4),
                                 nn.PReLU())

        self.layer2_res = self.SRGeneratorResBlock(device, gan_basis, num_blocks=16)

        self.layer3 = nn.Sequential(nn.Conv2d(gan_basis, gan_basis, kernel_size=3, padding=1),
                                    nn.PReLU())

        #self.layer3_res_last = self.SRGeneratorResBlock(device, gan_basis,num_blocks=1)

        self.layer4_px_shift = self.create_pixel_shuffle_block(gan_basis, num_blocks=2)
        self.layer5_conv = nn.Conv2d(gan_basis, 3, kernel_size=9, padding=4)
        #image output will be in the range of -1 and 1
        self.out_tanh=nn.Tanh()

    def forward(self, input):

        layer1_output = self.layer1(input)
        layer2_res_output = self.layer2_res(layer1_output)
        #Additinal  residual connection
        layer3_res_last_output = self.layer3(layer2_res_output)+ layer1_output
        layer4_pix_s_output = self.layer4_px_shift(layer3_res_last_output)
        layer5_conv_output = self.layer5_conv(layer4_pix_s_output)
        tanh_output = self.out_tanh(layer5_conv_output)
        return tanh_output

    def save_checkpoint(self,saved_dir, filename):
        # Save the weights to file
        torch.save(self.state_dict(), os.path.join(saved_dir, filename+".pth"))

    def load_from_checkpoint(self,saved_dir, filename):
        # Load the weights to file
        self.load_state_dict(torch.load(os.path.join(saved_dir, filename)))


class SRDiscriminator(nn.Module):
    '''
    Discriminator for SRGAN
    Built according to https://arxiv.org/abs/1609.04802
    With adjustments according to https://arxiv.org/pdf/1903.09922.pdf
    '''

    def create_sr_dicr_block(self, channels, num_blocks=3):
        #Sequence of Discrimnator basic blocks -Each is (conv-bn-lrelu)*2 (The first out of 2 does upsampling)
        out_channels=None
        self.discr_blocks = []
        for block in range(num_blocks):
            out_channels=(2*out_channels if out_channels is not None else channels)
            layers_block_list = [nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, stride=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
                                 nn.BatchNorm2d(out_channels),
                                 nn.LeakyReLU(0.2, inplace=True)]
            self.discr_blocks.append(nn.Sequential(*layers_block_list))
            channels=out_channels
        return nn.Sequential(*self.discr_blocks)

    def __init__(self, gan_basis=64, num_blocks=3):
        super(SRDiscriminator, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, gan_basis, kernel_size=3, padding=1),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(gan_basis, gan_basis, kernel_size=3, padding=1, stride=2),
                                    nn.BatchNorm2d(gan_basis),
                                    nn.LeakyReLU(0.2, inplace=True)
                                    )
        #3 discriminator basic blocks
        self.layer2_disc_blocks = self.create_sr_dicr_block(gan_basis, num_blocks=3)

        disc_block_num_outputs=gan_basis*(2**(num_blocks-1))
        self.layer3_dense_block=nn.Sequential( nn.Conv2d(disc_block_num_outputs, disc_block_num_outputs*2,kernel_size=3, padding=1),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(disc_block_num_outputs*2, 1, kernel_size=1))
        #I will end with logits and the loss function will do the sigmoid
        self.out=nn.Flatten()

    def forward(self, input):

        layer1_output = self.layer1(input)
        layer2_disc_output = self.layer2_disc_blocks(layer1_output)
        layer3_dense_output = self.layer3_dense_block(layer2_disc_output)
        output = self.out(layer3_dense_output)
        return output

    def save_checkpoint(self, saved_dir, filename):
        #Save the weights to file
        torch.save(self.state_dict(), os.path.join(saved_dir, filename + ".pth"))

    def load_from_checkpoint(self, saved_dir, filename):
        # Load the weights to file
        self.load_state_dict(torch.load(os.path.join(saved_dir, filename)))
