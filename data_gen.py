import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
import torchvision.transforms.functional as F

class DataGen(Dataset):
    def __init__(self, data_dir, transform_image=None):
        self.data_dir = data_dir
        self.transform_image = transform_image
        self.image_paths = [
            f"{data_dir}/{filename}"
            for filename in os.listdir(data_dir)
            if filename.endswith((".jpg", ".jpeg", ".png"))
        ]
        print(len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path).to(torch.float32) / 255

        try:
            image_split = image_path.split("/")
            partial_1 = [f"{image_split[i]}/" for i in range(len(image_split) - 1)]
            # print(partial_1)
            partial_1 = "".join(partial_1)
            partial_2 = image_split[-1]
            face_mask_path = f"{partial_1}masks/{partial_2[:-4]}_face_mask.jpg"
            #print(face_mask_path)
            face_mask = read_image(face_mask_path).to(torch.float32) / 255
            face_mask = face_mask
            face_mask = face_mask.expand(3, -1, -1)
            #print("found mask")
        except:
            #print("didn't found mask")
            face_mask = torch.ones_like(image)

        face_mask = face_mask.to("cuda")
        image = image.to("cuda")

        rotation_angle = 20 * random.random() - 10
        image = transforms.functional.rotate(image, rotation_angle)
        face_mask = transforms.functional.rotate(face_mask, rotation_angle)

        flip = random.random() >= 0.5

        if flip:
            image = transforms.functional.hflip(image)
            face_mask = transforms.functional.hflip(face_mask)

        alpha = random.uniform(0, 50)  # Controls the intensity of the transform
        sigma = random.uniform(3, 7)  # Controls the smoothness of the transform
        sigma = [sigma, sigma]
        alpha = [alpha, alpha]
        size = list(image.shape[-2:])
        dx = torch.rand([1, 1] + size) * 2 - 1
        if sigma[0] > 0.0:
            kx = int(8 * sigma[0] + 1)
            # if kernel size is even we have to make it odd
            if kx % 2 == 0:
                kx += 1
            dx = F.gaussian_blur(dx, [kx, kx], sigma)
        dx = dx * alpha[0] / size[0]

        dy = torch.rand([1, 1] + size) * 2 - 1
        if sigma[1] > 0.0:
            ky = int(8 * sigma[1] + 1)
            # if kernel size is even we have to make it odd
            if ky % 2 == 0:
                ky += 1
            dy = F.gaussian_blur(dy, [ky, ky], sigma)
        dy = dy * alpha[1] / size[1]
        displacement = torch.concat([dx, dy], 1).permute([0, 2, 3, 1])  # 1 x H x W x 2

        image = F.elastic_transform(image, displacement=displacement, fill=0)
        face_mask = F.elastic_transform(face_mask, displacement=displacement, fill=0)

        image = self.transform_image(image)

        image = transforms.Resize((256, 256))(image)
        face_mask = transforms.Resize((256, 256))(face_mask)

        output = torch.cat([image, face_mask], dim=0)

        return output


# Example usage:
dst_dir = "dst_d"
src_dir = "src_d"
batch_size = 8

transform_global = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image by up to 10 degrees
        transforms.Resize((256, 256)),
    ]
)

transform_image = transforms.Compose(
    [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)]
)

# Example usage:


# Create the DataGen object
dst = DataGen(dst_dir, transform_image=transform_image)

# Create a DataLoader to generate batches
dst = DataLoader(dst, batch_size=batch_size, shuffle=True)

src = DataGen(src_dir, transform_image=transform_image)
src = DataLoader(src, batch_size=batch_size, shuffle=True)


class InfiniteDataset:
    def __init__(self, dataset):
        super().__init__()
        self.ds = dataset
        self.iter = iter(self.ds)

    def __next__(self):
        try:
            output = next(self.iter)
        except:
            self.iter = iter(self.ds)
            output = next(self.iter)

        return output[:, :3, :, :], output[:, 3:, :, :]


src = InfiniteDataset(src)
dst = InfiniteDataset(dst)
'''
f, m = next(src)

f = f[0]
m = m[0]
print(f.shape, m.shape)

f = f.cpu().detach().transpose(0, 2)
m = m.cpu().detach().transpose(0, 2)

plt.imshow(f)

plt.show()

plt.imshow(m)

plt.show()
'''