import os
import json
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory


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
    os.environ["all_proxy"] = f"socks5://127.0.0.1:{port}"
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


def train_model(model, writer, savePath, dataloaders, device, criterion, optimizer, scheduler=None, num_epochs=25, returnBest=3):
    datasetsSize = {mode: len(dataloaders[mode].dataset)
                    for mode in ['train', 'valid']}

    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir) / Path('best.pth')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        nobestCount = 0

        for epoch in range(num_epochs):
            print('-' * 20)
            print(f'Epoch {epoch}/{num_epochs - 1}')
            
            # 每个epoch执行两个阶段 train 和 valid
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0 # 记录loss
                running_corrects = 0 # 记录分类正确数量

                for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                    optimizer.zero_grad()

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs) # (B, ClassPred)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 训练模型下反向传播
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 统计loss和acc
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()
                    
                epoch_loss = running_loss / datasetsSize[phase]
                epoch_acc = running_corrects / datasetsSize[phase]

                writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
                writer.add_scalar(f'Acc/{phase}', epoch_acc, epoch)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'valid' and scheduler is not None:
                    # 验证完再更新优化器
                    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
                    scheduler.step()

                # 保存高准确率模型
                if phase == 'valid' :
                    if epoch_acc > best_acc:
                        print(f"当前模型最佳，临时保存权重在{best_model_params_path.name}")
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                    else:
                        if best_acc - epoch_acc > 0.01:
                            nobestCount += 1
                        print(f"模型已有{nobestCount}个epoch没接近最佳Valid Acc: {best_acc:4f}!")
                        if nobestCount == returnBest:
                            model.load_state_dict(torch.load(best_model_params_path))
                            nobestCount = 0
                            print(f"模型已回退到最佳Valid ACC: {best_acc:4f}的权重！")

        # 读取最高准确率模型
        model.load_state_dict(torch.load(best_model_params_path))

    state: dict = {"model_state": model.state_dict(),
                   "optimizer": optimizer}

    torch.save(state, savePath / Path("best.pt"))
    return model

if __name__ == '__main__':
    # 设置权重下载目录
    TorchHome = Path("./Models")
    os.environ["TORCH_HOME"] = str(Path.cwd() / TorchHome)

    demo()
    
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    resnet18 = torchvision.models.resnet18(weights=weights)

    img = torch.randint(0, 256, size=(3, 320, 320), dtype=torch.float32)
    batch = img.unsqueeze(0)

    resnet18(batch)