import time
import torch
from utlis import *
import torch.nn as nn
from pathlib import Path
from torchvision import models
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR = 0.1 # Learning rate
L1 = 0.9 # Momentum
L2 = 1e-4 # Weight Decay
EPOCHS = 100
BATCHSIZE = 32
INIT_IMAGE_SIZE = (3, 320, 320) # 建议用dataloader里面的image来生成tensorboard的图记录
INIT_IMAGE = torch.zeros(INIT_IMAGE_SIZE, device=DEVICE).unsqueeze(0) # [3, 320, 320] -> [1, 3, 320, 320]
MILESTONES = list(map(int, [EPOCHS * 0.5, EPOCHS * 0.75]))

datasetPath = Path("./DataSets/ChestXRay2017_resize320")
outputPath = Path("./runs") / datasetPath.name
outputPath.mkdir(parents=True, exist_ok=True)

# 设置数据增强方法transform
data_transforms = {'train': transforms.Compose([transforms.RandomHorizontalFlip(0.2),  # 水平翻转
                                                transforms.RandomVerticalFlip(0.2), # 垂直翻转
                                                transforms.RandomRotation(3),  # 随机旋转
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),]),
                                                
                   'valid': transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])}

# 构建dataset、dataloader，使用字典分配 trian 和 valid
datasets = {mode: ChestRay2017(datasetPath , data_transforms[mode], (mode == "train"))
            for mode in ['train', 'valid']}
dataloaders = {mode: DataLoader(datasets[mode], batch_size=BATCHSIZE, shuffle=(mode == "train"))
               for mode in ['train', 'valid']}

class_names = getClassName(datasetPath)

model = models.resnet18(weights=None) # 构建模型

# model_state_dict = torch.load(Path("./Models/hub/checkpoints/resnet18-f37072fd.pth")) # 读取模型权重
# model.load_state_dict(model_state_dict, strict=True) # 严格匹配模型权重到模型中

fcIN = model.fc.in_features
model.fc = nn.Linear(fcIN, len(class_names)) 

model_ft = model.to(DEVICE)
model_ft.eval()
summary(model_ft, INIT_IMAGE_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ft.parameters(), 
                            lr=LR, momentum=L1, weight_decay=L2)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.1)


since = time.time()
strTime = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(since))
savePath = Path.cwd() / outputPath / Path(strTime)
Path.mkdir(savePath)
print(f"本次训练将会保存在{savePath}")

writer = SummaryWriter(log_dir=savePath)
writer.add_graph(model_ft, INIT_IMAGE)

model_ft = train_model(model_ft, writer, savePath, dataloaders, DEVICE, criterion, optimizer, scheduler, EPOCHS)
writer.close()

time_elapsed = time.time() - since
print(f'训练完成，用时：{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')