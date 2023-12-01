import torch
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter


def _saveModel(savePath: Path, 
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               ddp: bool = False,
               isBest: bool = True) -> None:
    '''保存模型'''
    state: dict = {'model_state': model.module.state_dict() if ddp else model.state_dict(),
                   'optimizer': optimizer}
    torch.save(state, savePath / (Path('best.pth') if isBest else Path('last.pth')))

def _cutmix_or_mixup(num_classes: int) -> v2.Transform:
    '''返回cutmix和mixup随机选择的transforms'''
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix = v2.CutMix(num_classes=num_classes)
    return v2.RandomChoice([cutmix, mixup])

def train(model: torch.nn.Module, 
          EXPRTIMENT: str, 
          savePath: Path, 
          dataloaders: dict, 
          criterion: torch.nn.modules.loss._WeightedLoss, 
          optimizer: torch.optim.Optimizer,
          scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None, 
          num_epochs: int = 100, 
          breakCount: int = 11, 
          cm: bool = True) -> torch.nn.Module:
    '''单机单卡训练'''

    writer = SummaryWriter(log_dir=savePath)
    writer.add_graph(model, dataloaders["valid"].dataset[0][0].unsqueeze(0).cuda())
    writer.add_text("Experiment Name", EXPRTIMENT)

    datasetsSize = {mode: len(dataloaders[mode].dataset)
                    for mode in ['train', 'valid']}
    
    class_names = dataloaders['train'].dataset.classes
    class_idxs = dataloaders['train'].dataset.class_to_idx

    cm_transform = _cutmix_or_mixup(len(class_names))
    
    # TensorBoard添加注释
    writer.add_text('dataset', str(datasetsSize), 0)
    writer.add_text('dataset', str(class_names), 1)
    writer.add_text('dataset', str(class_idxs), 2)
    writer.add_text('dataset', 'Train: ' + str(dataloaders['train'].dataset.statistics), 3)
    writer.add_text('dataset', 'Valid: ' + str(dataloaders['valid'].dataset.statistics), 4)

    best_acc = 0.0
    no_best_count = 0
    iter_count = {'train':0, 'valid':0}

    for epoch in range(num_epochs):
        print('-' * 20)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        
        # 每个epoch执行两个阶段 train 和 valid
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss: float = 0.0 # 记录loss
            running_corrects: int = 0 # 记录分类正确数量
            running_total:int = 0 # 记录所有数量

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                optimizer.zero_grad()

                inputs, labels = inputs.cuda(), labels.cuda()

                # 判断是否使用cutmix_or_mixup
                if phase == 'train' and cm == True:
                    _inputs, _labels = cm_transform(inputs, labels)
                else:
                    _inputs, _labels = inputs.clone(), labels.clone()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(_inputs) # (B, ClassPred)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, _labels)

                    # 训练模型下反向传播
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计loss和acc
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data).item()
                running_corrects += preds.eq(labels).sum().item()
                running_total += labels.numel()

                writer.add_scalars(f'Iteration/{phase}', {'Loss': running_loss/ running_total}, iter_count[phase])
                writer.add_scalars(f'Iteration/{phase}', {'Acc': running_corrects/ running_total}, iter_count[phase])

                iter_count[phase] += 1

            # dataloader的数据遍历完成，记录epoch的loss和acc
            epoch_loss = running_loss / datasetsSize[phase]
            epoch_acc = running_corrects / datasetsSize[phase]

            writer.add_scalars('Epoch/Loss', {phase: epoch_loss}, epoch)
            writer.add_scalars('Epoch/Acc', {phase: epoch_acc}, epoch)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 进入验证阶段末尾，更新学习率和保存最佳模型
            if phase == 'valid' :
                if scheduler is not None:
                    # 更新优化器参数
                    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
                    scheduler.step()

                if epoch_acc > best_acc:
                    # 记录最佳验证acc
                    no_best_count = 0
                    best_acc = epoch_acc
                    print(f"当前epoch模型的验证ACC {best_acc:.4f}最佳，开始保存模型...")
                    _saveModel(Path(savePath), model, optimizer)
                    
                else:
                    # 本次epoch的验证acc未达到最佳
                    if best_acc > epoch_acc:
                        no_best_count += 1
                        _saveModel(Path(savePath), model, optimizer, isBest=False)
                    if no_best_count == breakCount:
                        # 多次未达到最佳验证acc，读取最佳模型并保存在savePath后返回
                        model.load_state_dict(torch.load(Path(savePath) / Path('best.pth'))["model_state"])

                        print(f'模型已经{breakCount}个epoch非达到验证最佳{best_acc:.4f}！在{epoch}/{num_epochs - 1}提前结束训练。')
                        writer.close()
                        return model
    # 读取最高准确率模型
    print(f"加载验证最佳验证ACC {best_acc:.4f}的模型并返回...")
    model.load_state_dict(torch.load(Path(savePath) / Path('best.pth'))["model_state"])
    writer.close()
    return model

def ddptrain(model, writer, savePath, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, breakCount=1):
    pass