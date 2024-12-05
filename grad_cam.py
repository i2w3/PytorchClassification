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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

model = resnet50(weights=None)
model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 2))

model_state_dict = torch.load(Path("./runs/NEC/2024-10-10-15_30_20/best.pth"))["model_state"]  # 读取模型权重
model.load_state_dict(model_state_dict, strict=True)  # 严格匹配模型权重到模型中

transform = transforms.Compose([transforms.ToImage(),
                                # transforms.Resize([640,640]),
                                transforms.ToDtype(torch.float32, scale=True),
                                transforms.Normalize(mean=[0.43729363, 0.43703078, 0.43628642], std=[0.23020518, 0.23000101, 0.22991398])])
imagesPath = Path("/mnt/baode/YUNJIAO/ViP/DATA/NEC/test/necrotising_enterocolitis/0003821063.png")
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
# plt.show()
plt.savefig("./demo.png")