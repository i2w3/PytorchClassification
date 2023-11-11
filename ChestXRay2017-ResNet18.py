import time
import torch
from utlis import *
import torch.nn as nn
from pathlib import Path
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

EXPRTIMENT = "使用预训练权重训练"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR = 0.01  # Learning rate
MOMENTUM = 0.937  # Momentum
WEIGHT_DECAY = 5e-4  # Weight Decay、L2正则化

EPOCHS = 50
BATCHSIZE = 32
STEP_SIZE = 10

INIT_IMAGE_SIZE = (3, 320, 320)  # 建议用dataloader里面的image来生成tensorboard的图记录
INIT_IMAGE = torch.zeros(INIT_IMAGE_SIZE, device=DEVICE).unsqueeze(0)  # [3, 320, 320] -> [1, 3, 320, 320]

datasetPath = Path("./DataSets/ChestXRay2017_resize320")
outputPath = Path("./runs") / datasetPath.name
outputPath.mkdir(parents=True, exist_ok=True)

# 设置数据增强方法transform
data_transforms = {'train': transforms.Compose([transforms.RandomHorizontalFlip(),  # 水平翻转
                                                transforms.RandomRotation(15),  # 随机旋转
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ]),

                   'valid': transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])}

# 构建dataset、dataloader，使用字典分配 trian 和 valid
datasets = {mode: ChestRay2017Binary(datasetPath, data_transforms[mode], (mode == "train"))
            for mode in ['train', 'valid']}
dataloaders = {mode: DataLoader(datasets[mode], batch_size=BATCHSIZE, shuffle=(mode == "train"))
               for mode in ['train', 'valid']}

class_names = datasets["train"].classes
class_idxs = datasets["train"].class_to_idx

model = models.resnet18(weights=None)  # 构建模型
model_state_dict = torch.load(Path("./Models/hub/checkpoints/resnet18-f37072fd.pth"))  # 读取模型权重
model.load_state_dict(model_state_dict, strict=True)  # 严格匹配模型权重到模型中

# 重新训练resnet18的最后一个basic block 和 fc
for param in model.parameters():
    param.requires_grad = False  # 冻结所有层

for param in model.layer4.parameters():
    param.requires_grad = True  # 解冻最后两个basic block
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),
                            lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)

since = time.time()
strTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(since))
savePath = Path.cwd() / outputPath / Path(strTime)
Path.mkdir(savePath)
print(f"本次训练将会保存在{savePath}")

writer = SummaryWriter(log_dir=savePath)
writer.add_graph(model, INIT_IMAGE)
writer.add_text("Experiment Name", EXPRTIMENT)

model = train(model, writer, savePath, dataloaders, DEVICE, criterion, optimizer, scheduler, EPOCHS)
writer.close()

time_elapsed = time.time() - since
print(f'训练完成，用时：{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
