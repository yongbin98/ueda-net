import torch
import torch.nn as nn



class FiLM(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gamma = nn.Linear(1, features)
        self.beta = nn.Linear(1, features)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, stats):
        gamma = self.gamma(stats[:, 0:1]).unsqueeze(-1) + 1.0
        beta = self.beta(stats[:, 1:2]).unsqueeze(-1)
        return (x * gamma) + beta
    
class StudentConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, stride=2):
        super().__init__()

        p = ((k - 1) * dilation) // 2

        self.c1 = nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=p, groups=in_ch, padding_mode='replicate')
        self.c2 = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)
        self.gn1 = nn.GroupNorm(num_groups=out_ch//8, num_channels=out_ch)
        self.film = FiLM(out_ch)
        self.relu = nn.LeakyReLU(0.2)

        # --- Shortcut ---
        self.shortcut = nn.Sequential()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=out_ch//8, num_channels=out_ch)
            )

    def forward(self, x, stats):
        residual = self.shortcut(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.gn1(x)
        out = self.film(x, stats)
        return self.relu(out + residual)

class StudentDeconvBlock2(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, k, stride=2, padding=1, output_padding=1):
        super().__init__()

        self.upsample = nn.ConvTranspose1d(
            in_ch, in_ch // 2, kernel_size=k, stride=stride,
            padding=padding, output_padding=output_padding
        )

        combined_ch = (in_ch // 2) + skip_ch

        self.convs = nn.Sequential(
            nn.Conv1d(combined_ch, out_ch, kernel_size=k, padding=(k-1)//2, padding_mode='replicate'),
            nn.GroupNorm(num_groups=out_ch//8, num_channels=out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)

        return self.convs(x)


class Student_Youngbin(nn.Module):
    def __init__(self):
        super().__init__()
        ks = 5

        self.e1 = StudentConvBlock(1,   8, k=ks, stride=2, dilation=1)
        self.e2 = StudentConvBlock(8,  16, k=ks, stride=2, dilation=1)
        self.e3 = StudentConvBlock(16,  32, k=ks, stride=2, dilation=1)
        self.e4 = StudentConvBlock(32,  64, k=ks, stride=2, dilation=1)
        self.e5 = StudentConvBlock(64,  128, k=ks, stride=2, dilation=1)

        p_up = (ks-1)//2 # Padding for k=5
        self.d1 = StudentDeconvBlock2(128, 64, 64, k=ks, stride=2, padding=p_up, output_padding=1)
        self.d2 = StudentDeconvBlock2(64, 32, 32, k=ks, stride=2, padding=p_up, output_padding=1)
        self.d3 = StudentDeconvBlock2(32, 16, 16, k=ks, stride=2, padding=p_up, output_padding=1)
        self.d4 = StudentDeconvBlock2(16, 8, 8, k=ks, stride=2, padding=p_up, output_padding=1)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(8, 8, kernel_size=ks, stride=2, padding=p_up, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, kernel_size=ks, padding=p_up, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 1, kernel_size=ks, padding=p_up, padding_mode='replicate')
        )

    def forward(self, x, stats, return_bottleneck=False):
        # Encoder
        x1 = self.e1(x, stats)   # 16,  L/2
        x2 = self.e2(x1, stats)  # 32,  L/4
        x3 = self.e3(x2, stats)  # 64,  L/8
        x4 = self.e4(x3, stats)  # 128, L/16
        x5 = self.e5(x4, stats)  # 128, L/16

        y = self.d1(x5, x4)
        y = self.d2(y, x3)
        y = self.d3(y, x2)
        y = self.d4(y, x1)
        y = self.final_layer(y)
        if return_bottleneck:
            return y, x1, x2, x3, x4, x5
        return y

class BottleneckAdapter(nn.Module):
    def __init__(self, student_ch=128, teacher_ch=256):
        super().__init__()
        # 1x1 Conv to match channels
        self.project = nn.Conv1d(student_ch, teacher_ch, kernel_size=1)

    def forward(self, x):
        return self.project(x)
