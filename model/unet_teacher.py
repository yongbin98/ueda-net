import torch
import torch.nn as nn

# FFN Block
class FFNBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, padding_mode='replicate')
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.attn_block = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FFNBlock(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, L) → (B, L, C)

        residual = x
        attn_out, _ = self.attn_block(x, x, x)
        x = attn_out + residual
        x = self.norm1(x)

        residual = x
        x = x.permute(0, 2, 1)          # → (B, C, L) for Conv1D
        x = self.ffn(x)
        x = x.permute(0, 2, 1)          # → (B, L, C)
        x = x + residual
        x = self.norm2(x)

        return x.permute(0, 2, 1)


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

class ConvBlock_YB(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, dilation=1, stride=2):
        super().__init__()

        p_dilated = ((k - 1) * dilation) // 2
        p_standard = (k - 1) // 2

        self.c1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p_dilated, dilation=dilation, padding_mode='replicate')
        self.gn1 = nn.GroupNorm(num_groups=max(1, out_ch//8), num_channels=out_ch)

        self.film = FiLM(out_ch)
        self.act1 = nn.LeakyReLU(0.2)

        self.c2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=p_standard, stride=stride, padding_mode='replicate')
        self.gn2 = nn.GroupNorm(num_groups=max(1, out_ch//8), num_channels=out_ch)

        self.final_act = nn.LeakyReLU(0.2)

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
        x = self.gn1(x)
        x = self.film(x, stats) # Inject stats here
        x = self.act1(x)

        x = self.c2(x)
        x = self.gn2(x)

        return self.final_act(x + residual)

class DeconvBlock_YB(nn.Module):
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


class Denoising_Youngbin(nn.Module):
    def __init__(self):
        super().__init__()
        ks = 5

        self.e1 = ConvBlock_YB(1,   16, k=ks, stride=2)
        self.e2 = ConvBlock_YB(16,  32, k=ks, stride=2)
        self.e3 = ConvBlock_YB(32,  64, k=ks, stride=2)
        self.e4 = ConvBlock_YB(64,  128, k=ks, stride=2)
        self.e5 = ConvBlock_YB(128,  256, k=ks, stride=2)

        self.tf = TransformerBlock()

        p_up = (ks - 1) // 2 # Padding for k=5
        self.d1 = DeconvBlock_YB(256, 128, 128, k=ks, stride=2, padding=p_up, output_padding=1)
        self.d2 = DeconvBlock_YB(128, 64, 64, k=ks, stride=2, padding=p_up, output_padding=1)
        self.d3 = DeconvBlock_YB(64,  32, 32, k=ks, stride=2, padding=p_up, output_padding=1)
        self.d4 = DeconvBlock_YB(32,  16, 16, k=ks, stride=2, padding=p_up, output_padding=1)


        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(16, 16, kernel_size=ks, stride=2, padding=p_up, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 16, kernel_size=ks, padding=p_up, padding_mode='replicate'),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, kernel_size=ks, padding=p_up, padding_mode='replicate')
        )

    def forward(self, x, stats, return_bottleneck=False):
        # Encoder
        x1 = self.e1(x, stats)   # 16,  L/2
        x2 = self.e2(x1, stats)  # 32,  L/4
        x3 = self.e3(x2, stats)  # 64,  L/8
        x4 = self.e4(x3, stats)  # 128, L/16
        x5 = self.e5(x4, stats)

        x5 = self.tf(x5)

        y = self.d1(x5, x4)
        y = self.d2(y, x3)
        y = self.d3(y, x2)
        y = self.d4(y, x1)
        y = self.final_layer(y)
        if return_bottleneck:
            return y, x1, x2, x3, x4, x5
        return y
