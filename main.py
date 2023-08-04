import os
from PIL import Image
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from data.base_dataset import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from networks import U2NET

device = "cuda"

checkpoint_path = '/root/autodl-tmp/cloth_segm_u2net_latest.pth'


transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transform_rgb = transforms.Compose(transforms_list)

net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device)
net = net.eval()
print('模型加载完毕')

# 单个图片的分割
def segmentation(image: Image.Image):
    original_image = image.copy()
    image = image.convert("RGB")
    image_tensor = transform_rgb(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)

    output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    # 将输出的蒙版图像转换为Image对象
    Image.fromarray(output_arr.astype("uint8"), mode="L").save("result.png")

if __name__ == "__main__":
    image = Image.open("test.jpg")
    segmentation(image)