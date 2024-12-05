import os
import json
import torch
import torchvision
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def tensor2Image(tensor, isNormalize=True):
    """将 torch.Tensor 转为 numpy 并反标准化"""
    image = tensor.numpy().transpose((1, 2, 0))
    if isNormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
    return image


def showImage(*args):
    """显示 torch.Tensor 在 PLT 上"""
    plt.figure(figsize = (5 * len(args), 5))
    for i, arg in enumerate(args):
        ax = plt.subplot(1, len(args), i+1) # 生成 1行 len(args) 列 的子图，并在 i+1处画图
        plt.imshow(tensor2Image(arg, i != 0))
        ax.set_title(f"Shape: {arg.shape}")
    plt.tight_layout()
    plt.show()

def getClassName(folder:Path, jsonFile = Path("label.json")):
    """从json文件中获取类名"""
    with open(folder / jsonFile, 'r') as f:
        data = json.load(f)
    keys = data.keys()
    return list(keys)

def proxy(port="7890"):
    """为git添加临时代理"""
    os.environ["all_proxy"] = f"socks5://127.0.0.1:{port+1}"
    os.environ["http_proxy"] = f"http://127.0.0.1:{port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{port}"
    

def getModelList():
    print("所有的预训练模型可此查询：https://pytorch.org/vision/stable/models.html")
    Classification_models = torchvision.models.list_models(module=torchvision.models)
    SemanticSegmentation_models = torchvision.models.list_models(module=torchvision.models.segmentation)
    print(f"预训练的分类模型有：{Classification_models}")
    print(f"预训练的分割模型有:{SemanticSegmentation_models}")


def demo():
    # eg1:无预训练的模型
    resnet50 = torchvision.models.resnet50(weights=None)

    # eg2:有预训练的模型，字符串调用，使用IMAGENET1K_V2(ACC:80.858%)，可选：IMAGENET1K_V1(ACC:76.130%)
    resnet50 = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    
    # eg3:用weights调用预训练权重并加载到模型中
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 # 定义权重文件，还没下载
    resnet50 = torchvision.models.resnet50(weights=weights) # 不是字符串调用了
    # 这样调用可以查看该权重使用的tranforms方法，还有类别
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    print(preprocess)
    print(categories)

    img = torch.randint(0, 256, size=(3, 320, 320), dtype=torch.uint8) # 这里是为了生成图像，严格来说神经网络量化前需要输入的torch.float32
    preprocess_img = preprocess(img)
    showImage(img, preprocess_img)

    batch = preprocess(img).unsqueeze(0) # 将transforms后的数据封装为一个batch

    prediction = resnet50(batch).squeeze(0).softmax(0) # 数据输入模型[batchSize, ClassNum] -> 取出batch[ClassNum] -> 维度0激活

    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score}%")


if __name__ == '__main__':
    # 设置权重下载目录
    TorchHome = Path("./Models")
    os.environ["TORCH_HOME"] = str(Path.cwd() / TorchHome)

    # demo()
    
    # weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    # resnet18 = torchvision.models.resnet18(weights=weights)

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
    resnet50 = torchvision.models.resnet50(weights=weights)

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    resnet50 = torchvision.models.resnet50(weights=weights)

    # weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
    # densenet121 = torchvision.models.densenet121(weights=weights)

    # weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    # vitb16 = torchvision.models.vit_b_16(weights=weights)

    # weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
    # vitb16 = torchvision.models.vit_b_16(weights=weights)

    # weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
    # vitb16 = torchvision.models.vit_b_16(weights=weights)
