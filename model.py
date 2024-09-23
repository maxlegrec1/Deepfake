import torch
from torch import nn
from torch.nn import functional as F


class Downscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, *kwargs):
        super().__init__(*kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(
            self.in_ch,
            self.out_ch,
            kernel_size=self.kernel_size,
            stride=2,
            padding=2,
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

    def get_out_ch(self):
        return self.out_ch


class Upscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, *kwargs):
        super().__init__(*kwargs)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(
            self.in_ch,
            self.out_ch*4,
            kernel_size=self.kernel_size,
            padding="same",
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x,0.1)
        x = depth_to_space(x,size = 2)
        return x

    def get_out_ch(self):
        return self.out_ch


class Encoder(nn.Module):
    def __init__(self, img_ch=3, enc_ch=512, *kwargs):
        super().__init__(*kwargs)
        self.in_ch = img_ch
        self.enc_ch = enc_ch
        self.enc_1 = Downscale(self.in_ch, self.enc_ch // 8)
        self.enc_2 = Downscale(self.enc_ch // 8, self.enc_ch // 4)
        self.enc_3 = Downscale(self.enc_ch // 4, self.enc_ch // 2)
        self.enc_4 = Downscale(self.enc_ch // 2, self.enc_ch)

    def forward(self, x):

        x = F.leaky_relu(self.enc_1(x), 0.1)
        x = F.leaky_relu(self.enc_2(x), 0.1)
        x = F.leaky_relu(self.enc_3(x), 0.1)
        x = F.leaky_relu(self.enc_4(x), 0.1)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size, padding="same")

    def forward(self, inp):
        x = self.conv1(inp)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x = F.leaky_relu(inp + x, 0.2)
        return x


class Decoder(nn.Module):
    def __init__(self, img_ch=3, enc_ch=1024, d_mask_ch=22, *kwargs):
        super().__init__(*kwargs)
        self.in_ch = img_ch
        self.enc_ch = enc_ch
        self.enc_1 = Upscale(self.enc_ch, self.enc_ch // 2)
        self.res1 = ResidualBlock(self.enc_ch // 2, kernel_size=3)
        self.enc_2 = Upscale(self.enc_ch // 2, self.enc_ch // 2)
        self.res2 = ResidualBlock(self.enc_ch // 2, kernel_size=3)
        self.enc_3 = Upscale(self.enc_ch // 2, self.enc_ch // 4)
        self.res3 = ResidualBlock(self.enc_ch // 4, kernel_size=3)
        self.final_conv = nn.Conv2d(
            self.enc_ch // 4,
            3,
            kernel_size=1,
            padding="same",
        )
        self.final_conv1 = nn.Conv2d(
            self.enc_ch // 4,
            3,
            kernel_size=3,
            padding="same",
        )   
        self.final_conv2 = nn.Conv2d(
            self.enc_ch // 4,
            3,
            kernel_size=3,
            padding="same",
        )          
        self.final_conv3 = nn.Conv2d(
            self.enc_ch // 4,
            3,
            kernel_size=3,
            padding="same",
        )   

        self.upscalem0 = Upscale(self.enc_ch, d_mask_ch * 8)
        self.res1m = ResidualBlock(d_mask_ch * 8, kernel_size=3)
        self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 4)
        self.res2m = ResidualBlock(d_mask_ch * 4, kernel_size=3)
        self.upscalem2 = Upscale(d_mask_ch * 4, d_mask_ch * 2)
        self.res3m = ResidualBlock(d_mask_ch * 2, kernel_size=3)
        self.upscalem3 = Upscale(d_mask_ch * 2, d_mask_ch * 1)
        self.out_convm = nn.Conv2d(d_mask_ch * 1, 1, kernel_size=1, padding="same")

    def forward(self, x):
        m = x
        x = self.enc_1(x)
        x = self.res1(x)
        x = self.enc_2(x)
        x = self.res2(x)
        x = self.enc_3(x)
        x = self.res3(x)
        x = depth_to_space(torch.cat([self.final_conv(x),self.final_conv1(x),self.final_conv2(x),self.final_conv3(x)], dim = 1),2)
        x = F.sigmoid(x)

        m = self.upscalem0(m)
        m = self.res1m(m)
        m = self.upscalem1(m)
        m = self.res2m(m)
        m = self.upscalem2(m)
        m = self.res3m(m)
        m = self.upscalem3(m)
        m = F.sigmoid(self.out_convm(m))
        m = m.expand(-1, 3, -1, -1)

        return x, m


class Inter(nn.Module):
    def __init__(self):
        super().__init__()
        self.interAB_1 = nn.Linear(512 * 16 * 16, 256)
        self.interAB_2 = nn.Linear(256, 512 * 8 * 8)
        self.conv_AB = Upscale(512,512)
        self.interB_1 = nn.Linear(512 * 16 * 16, 256)
        self.interB_2 = nn.Linear(256, 512 * 8 * 8)
        self.conv_B = Upscale(512,512)
        self.interA_1 = nn.Linear(512 * 16 * 16, 256)
        self.interA_2 = nn.Linear(256, 512 * 8 * 8)
        self.conv_A = Upscale(512,512)


        self.srcmat = torch.nn.Parameter(torch.empty((1,256),device="cuda"))
        self.dstmat = torch.nn.Parameter(torch.empty((1,256),device="cuda"))



    def forward(self, x):
        x_AB = self.interAB_1(x.view(x.shape[0], -1))
        x_AB = self.interAB_2(x_AB)
        x_AB = x_AB.view(-1,512,8,8)
        x_AB = self.conv_AB(x_AB)
        x_B = self.interB_1(x.view(x.shape[0], -1))
        x_B = self.interB_2(self.dstmat.expand(x.shape[0],-1)+ x_B)
        x_B = x_B.view(-1,512,8,8)
        x_B = self.conv_B(x_B)
        x_A = self.interA_1(x.view(x.shape[0], -1))
        x_A = self.interA_2(self.srcmat.expand(x.shape[0],-1)+ x_A)
        x_A = x_A.view(-1,512,8,8)
        x_A = self.conv_A(x_A)

        recon_x_src = torch.cat([x_AB, x_A], dim=1)
        recon_x_dst = torch.cat([x_AB, x_B], dim=1)

        return recon_x_src, recon_x_dst
def depth_to_space(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    Rearrange depth data into spatial data.
    
    Args:
        x (torch.Tensor): Input tensor in NCHW format.
        size (int): Block size for rearrangement.
    
    Returns:
        torch.Tensor: Rearranged tensor in NCHW format.
    """
    b, c, h, w = x.size()
    oh, ow = h * size, w * size
    oc = c // (size * size)

    # Reshape and permute
    x = x.view(b, size, size, oc, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.contiguous().view(b, oc, oh, ow)

    return x

class DFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.inter = Inter()
        self.dec = Decoder()

    def forward(self, x):

        x = self.enc(x)
        latent_x_src, latent_x_dst = self.inter(x)

        recon_x_src, recon_x_src_mask = self.dec(latent_x_src)
        recon_x_dst, recon_x_dst_mask = self.dec(latent_x_dst)
        return recon_x_src, recon_x_src_mask, recon_x_dst, recon_x_dst_mask


model = DFM()
