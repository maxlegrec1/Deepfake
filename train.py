import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_gen import dst, src
from dssim import dssim, gaussian_blur
from model import DFM

writer = SummaryWriter()


model = DFM().to("cuda")

mse_loss = torch.nn.MSELoss()


def compute_loss(img_true, img_pred, mask_true, mask_pred):

    mask_blurred = gaussian_blur(mask_true, radius=8)

    mse = (mask_blurred * (img_true - img_pred) ** 2).mean()
    # m_ssim = 1 - ssim_module(img_true * mask_true, img_pred * mask_true)
    m_ssim = (
        dssim(
            img_true * mask_blurred,
            img_pred * mask_blurred,
            filter_size=int(256 / 11.6),
        )
        + dssim(
            img_true * mask_blurred,
            img_pred * mask_blurred,
            filter_size=int(256 / 23.2),
        )
    ).mean()
    mask_mse = ((mask_true[:,0,:,:] - mask_pred[:,0,:,:]) ** 2).mean()

    return 10 * mse + 5 * m_ssim + 10 * mask_mse


opt = torch.optim.Adam(model.parameters(), lr=5e-5)

total_steps = 50_000

for step in tqdm(range(total_steps)):

    src_img, src_mask = next(src)
    dst_img, dst_mask = next(dst)

    src_pred, src_mask_pred, _, _ = model(src_img)
    _, _, dst_pred, dst_mask_pred = model(dst_img)

    deepfake, _, true_face, _ = model(dst_img)
    merged = dst_img * (1-dst_mask) + deepfake * dst_mask
    deepfake = torch.cat([src_img, src_pred,dst_img, true_face, deepfake, merged], dim=0)
    if step % 100 == 0:  # Adjust the frequency as needed
        torchvision.utils.save_image(deepfake, f"run_images/deepfake_{step}.png")

    loss_src = compute_loss(src_img, src_pred, src_mask, src_mask_pred)

    loss_dst = compute_loss(dst_img, dst_pred, dst_mask, dst_mask_pred)

    loss = loss_src + loss_dst

    loss.backward()
    writer.add_scalar("Loss/train", loss, step)
    print(f"loss : {loss.item()}")
    opt.step()
    opt.zero_grad()
