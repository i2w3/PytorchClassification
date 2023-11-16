import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from torchvision.transforms import v2 as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = resnet50(weights=None)
model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 2))

model_state_dict = torch.load(Path("./runs/NEC_resize320/2023-11-16-09_48_04/best.pt"))["model_state"]  # 读取模型权重
model.load_state_dict(model_state_dict, strict=True)  # 严格匹配模型权重到模型中

transform = transforms.Compose([transforms.ToImage(),
                                transforms.ToDtype(torch.float32, scale=True),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
imagesPath = Path("./DataSets/NEC_resize320/valid/NEC/0001_000001_1.2.392.200046.100.2.1.97028795841.121214164031.1.1.1.png")
image = Image.open(imagesPath).convert("RGB")
rgb_img = np.array(image)
rgb_img = np.float32(rgb_img) / 255


target_layers = [model.layer4[-1]]
input_tensor = transform(image).unsqueeze(0)

with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
    targets = [ClassifierOutputTarget(1)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()