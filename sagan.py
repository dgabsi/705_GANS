import  torch
import torch.nn as nn
import os
import torch.nn.functional as F



class Attention(nn.Module):
    #Attention block of Self-Attention GAN. According to https://arxiv.org/abs/1805.08318.
    #3 steps : matrix multiplication-softamx(scaling)- matrix multiplication

    def __init__(self, in_channels, attention_dim_div=8):
        super(Attention, self).__init__()

        self.keys = nn.Conv2d(in_channels, in_channels // attention_dim_div, kernel_size=1)
        self.queries = nn.Conv2d(in_channels, in_channels // attention_dim_div, kernel_size=1)
        #self.h = nn.Conv2d(in_channels, in_channels// attention_dim_div, kernel_size=1)
        self.values = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        queries_outputs = torch.flatten(self.queries(input), start_dim=2)  # (B,C/8,  H*W)
        keys_outputs = torch.flatten(self.keys(input), start_dim=2)  # (B,C/8,  H*W)
        values_outputs = torch.flatten(self.values(input), start_dim=2)  # (B,C,  H*W)
        #h_outputs = torch.flatten(self.h(input), start_dim=2)#(B,C/8,  H*W)

        # first attention multiplication queries * keys
        attn_mul1 = torch.bmm(torch.transpose(queries_outputs, 1, 2), keys_outputs)

        # normaliing the attention with the dum of 1 (using softmax)
        attn_scaled = F.softmax(attn_mul1, dim=-1)

        # second multiplication attention: values *scaled attention
        attn_mul2 = torch.bmm( values_outputs, torch.transpose(attn_scaled, 1, 2))

        #attention_output=self.values(attn_mul2 )
        # back to the input size
        attention_output = attn_mul2.view(input.size())

        # multiplication by a learnable parameter gamma plus a residual link to input
        output = self.gamma * attention_output + input

        return output


class SAGeneratorBlock(nn.Module):
    #Generator block according to https://arxiv.org/abs/1805.08318. It is based on DCGAN but with spectral norm and attention if with_attn=True
    def __init__(self, in_channels, with_attn=False, attention_div=8, last=False):
        super(SAGeneratorBlock, self).__init__()

        self.last = last
        self.with_attn = with_attn

        if not last:
            out_channels = int(in_channels // 2)
            self.conv = nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
            if with_attn:
                self.attention = Attention(out_channels, attention_div)

        else:
            out_channels = 3
            self.conv = nn.utils.spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2,
                                       padding=1, bias=False))

    def forward(self, inputs):

        conv_out = self.conv(inputs)
        out = conv_out

        if not self.last:
            bn_out = self.bn(conv_out)
            relu_out = self.relu(bn_out)
            out = relu_out
            if self.with_attn:
                attn_out = self.attention(relu_out)
                out = attn_out
        return out



class SAGenerator(nn.Module):
    # Generator block according to https://arxiv.org/abs/1805.08318. It is based on DCGAN but with spectral norm and attention in specific layer

    def __init__(self, noise_size, image_size):
        super(SAGenerator, self).__init__()

        # The default structure of dcgan is built on 64 hidden size. starting from 4*4 image, there are 4 layers, each multiplies the size to by 2,
        # while the number of channels start from hidden size*16 and is divided each layer
        # If the image size is a different mutiple of 2 ,the hidden size is adjusted accrodingly since its a fully convolutinal cnn
        default_hidden_size = 64
        hidden_size_divider = (image_size // default_hidden_size)
        hidden_size = int(default_hidden_size // hidden_size_divider)

        self.noise_size = noise_size

        self.convt1 = self.input_conv = nn.Sequential(nn.ConvTranspose2d(noise_size, hidden_size * 16, kernel_size=4, stride=1),
                                                      nn.ReLU())
        self.convt2 = SAGeneratorBlock(hidden_size * 16)
        self.convt3 = SAGeneratorBlock(hidden_size* 8, with_attn=True)
        self.convt4 = SAGeneratorBlock(hidden_size * 4)
        self.convt5 = SAGeneratorBlock(hidden_size * 2, last=True)
        self.tanh = nn.Tanh()

        #DCGAN Initalization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
            if isinstance(layer, (nn.BatchNorm2d)):
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0)


    def forward(self, noise_inputs):

        convt1_output = self.convt1(noise_inputs)
        convt2_output = self.convt2(convt1_output)
        convt3_output = self.convt3(convt2_output)
        convt4_output = self.convt4(convt3_output)
        convt5_output = self.convt5(convt4_output)
        output = self.tanh(convt5_output)

        return output

    def generate_noise(self, batch_size):
        #Noise from randn distribution
        noise = torch.randn(batch_size, self.noise_size, 1, 1)

        return noise

    def save_checkpoint(self,saved_dir, filename):
        # Save the weights to file
        torch.save(self.state_dict(), os.path.join(saved_dir, filename+".pth"))

    def load_from_checkpoint(self,saved_dir, filename):
        # Load the weights from file
        self.load_state_dict(torch.load(os.path.join(saved_dir, filename)))


class SADiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, last=False, with_attn=False):
        super(SADiscriminatorBlock, self).__init__()

        self.last = last
        self.with_attn = with_attn

        if not last:
            out_channels = 2 * in_channels
            padding = 1
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=padding, bias=False))
            self.bn = nn.BatchNorm2d(out_channels)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2)
            if with_attn:
                self.attention = Attention(out_channels)
        else:
            out_channels = 1
            padding = 0
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=padding))


    def forward(self, inputs):
        conv_out = self.conv(inputs)
        out = conv_out

        if not self.last:
            bn_out = self.bn(conv_out)
            lrelu_out = self.lrelu(bn_out)
            out = lrelu_out
            if self.with_attn:
                attn_out = self.attention(lrelu_out)
                out = attn_out

        return out


class SADiscriminator(nn.Module):


    def __init__(self, image_size):
        super(SADiscriminator, self).__init__()

        # It is assumed image size is a multiplier power of 2 and is grater or equal to 64
        # The default structure of dcgan is built on 64 hidden size. There are 4 layers, each multiplies the channels by 2 and divided the spatial size by 2
        # Mirror of the Generator
        default_hidden_size = 64
        hidden_size_divider = (image_size // default_hidden_size)
        hidden_size = int(default_hidden_size // hidden_size_divider)

        self.conv1 = self.input_conv = nn.Sequential(nn.Conv2d(3, hidden_size, kernel_size=4, stride=2, padding=1),
                                        nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = SADiscriminatorBlock( hidden_size, with_attn=True)
        self.conv3 = SADiscriminatorBlock(hidden_size * 2)
        self.conv4 = SADiscriminatorBlock(hidden_size * 4)
        self.conv5 = SADiscriminatorBlock(hidden_size * 8, last=True)

        # DCGAN Initalization
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
            if isinstance(layer, (nn.BatchNorm2d)):
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant_(layer.bias.data, 0)

    def forward(self, inputs):

        conv1_output = self.conv1(inputs)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        output = conv5_output
        return output

    def save_checkpoint(self, saved_dir, filename):
        # Save the weights to file
        torch.save(self.state_dict(), os.path.join(saved_dir, filename + ".pth"))

    def load_from_checkpoint(self, saved_dir, filename):
        # Load the weights from file
        self.load_state_dict(torch.load(os.path.join(saved_dir, filename)))