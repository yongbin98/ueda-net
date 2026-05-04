import torch
import torch.nn as nn

class Conv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Conv_block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride = stride)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.LeakyReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

## Deconvolution block
class deconv_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, out_padding, stride):
        super(deconv_block,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.out_padding = out_padding
        self.deconv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=padding, output_padding = out_padding, stride = stride)
        self.padding = padding
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class denoising_model(nn.Module):
  def __init__(self):
    super(denoising_model,self).__init__()
    self.conv1_layer = Conv_block(1, 16, 65, 32,2)
    self.conv2_layer = Conv_block(16, 32, 31, 15, 2)
    self.conv3_layer = Conv_block(32, 64, 15, 7, 2)
    self.conv4_layer = Conv_block(64, 128, 7, 3, 2)
    self.conv5_layer = Conv_block(128, 256, 7, 3, 2)
    self.deconv1= deconv_block(256, 128, 7, 3, 1, 2)
    self.deconv2 = deconv_block(128, 64, 7, 3, 1, 2)
    self.deconv3 = deconv_block(64, 32, 15, 7, 1, 2)
    self.deconv4 = deconv_block(32, 16, 31, 15, 1, 2)
    self.deconv5 = torch.nn.ConvTranspose1d(16, 1, 7, padding=3, output_padding = 1, stride = 2)

  def forward(self,x):
    x1 = self.conv1_layer(x)
    x2 = self.conv2_layer(x1)
    x3 = self.conv3_layer(x2)
    x4 = self.conv4_layer(x3)
    x5 =  self.conv5_layer(x4)
    x = self.deconv1(x5)
    x = self.deconv2(x+x4)
    x = self.deconv3(x+x3)
    x = self.deconv4(x+x2)
    x = self.deconv5(x+x1)
    return x
