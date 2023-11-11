import torch
from tqdm import tqdm
from pathlib import Path
from tempfile import TemporaryDirectory


def train(model, writer, savePath, dataloaders, device, criterion, optimizer, scheduler=None, num_epochs=25, breakCount=11):
    datasetsSize = {mode: len(dataloaders[mode].dataset)
                    for mode in ['train', 'valid']}
    
    class_names = dataloaders['train'].dataset.classes
    class_idxs = dataloaders['train'].dataset.class_to_idx

    writer.add_text("dataset", str(datasetsSize), 0)
    writer.add_text("dataset", str(class_names), 1)
    writer.add_text("dataset", str(class_idxs), 2)
    writer.add_text("dataset", str(dataloaders['train'].dataset.statistics), 3)
    writer.add_text("dataset", str(dataloaders['valid'].dataset.statistics), 3)
    

    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir) / Path('best.pth')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        nobestCount = 0
        iter_count = {"train":0, "valid":0}

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
                running_total = 0

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
                    running_total += labels.size(0)

                    writer.add_scalars(f'Iteration/{phase}', {"Loss": running_loss/ running_total}, iter_count[phase])
                    writer.add_scalars(f'Iteration/{phase}', {"Acc": running_corrects/ running_total}, iter_count[phase])

                    iter_count[phase] += 1
                    
                epoch_loss = running_loss / datasetsSize[phase]
                epoch_acc = running_corrects / datasetsSize[phase]

                writer.add_scalars("Epoch/Loss", {phase: epoch_loss}, epoch)
                writer.add_scalars("Epoch/Acc", {phase: epoch_acc}, epoch)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # 进入验证阶段末尾，更新学习率和保存最佳模型
                if phase == 'valid' :
                    if scheduler is not None:
                        # 验证完再更新优化器
                        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
                        scheduler.step()

                    if epoch_acc > best_acc:
                        print(f"当前模型最佳，临时保存权重在{best_model_params_path.name}")
                        nobestCount = 0
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                    else:
                        if best_acc - epoch_acc > 0.01:
                            nobestCount += 1
                        if nobestCount == breakCount:
                            model.load_state_dict(torch.load(best_model_params_path))

                            state: dict = {"model_state": model.state_dict(),
                                           "optimizer": optimizer}

                            torch.save(state, savePath / Path("best.pt"))
                            
                            print(f"模型已经{breakCount}个epoch非达到验证最佳！在{epoch}/{num_epochs - 1}提前结束训练。")

                            return model

        # 读取最高准确率模型
        model.load_state_dict(torch.load(best_model_params_path))

    state: dict = {"model_state": model.state_dict(),
                   "optimizer": optimizer}

    torch.save(state, savePath / Path("best.pt"))
    return model