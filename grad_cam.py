import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.models import resnet18, resnet50
from torchvision.transforms import v2 as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = resnet18(weights=None)
model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 3))

model_state_dict = torch.load(Path("./runs/ChestXRay2017_resize320/2023-11-23-14_28_09/best.pt"))["model_state"]  # 读取模型权重
model.load_state_dict(model_state_dict, strict=True)  # 严格匹配模型权重到模型中

transform = transforms.Compose([transforms.ToImage(),
                                transforms.ToDtype(torch.float32, scale=True),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
imagesPath = Path("./DataSets/ChestXRay2017_resize320/valid/PNEUMONIA/person1_virus_6.jpeg")
image = Image.open(imagesPath).convert("RGB")
rgb_img = np.array(image)
rgb_img = np.float32(rgb_img) / 255


target_layers = [model.layer4[-1]]
input_tensor = transform(image).unsqueeze(0)

with GradCAM(model=model, target_layers=target_layers, use_cuda=True) as cam:
    targets = [ClassifierOutputTarget(2)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.show()
