import torch
from tqdm import tqdm
from pathlib import Path
from tempfile import TemporaryDirectory

def trainStepbyStep(model, writer, savePath, dataloaders, device, criterion, optimizer, scheduler=None, num_epochs=25, returnBest=3):
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

            model.train()
            running_loss = 0.0 # 记录loss
            running_corrects = 0 # 记录分类正确数量

            for inputs, labels in tqdm(dataloaders["train"], desc="train"):
                optimizer.zero_grad()

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs) # (B, ClassPred)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # 统计loss和acc
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / datasetsSize["train"]
            epoch_acc = running_corrects / datasetsSize["train"]

            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Acc/train', epoch_acc, epoch)
            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            with torch.no_grad():
                model.eval()
                running_loss = 0.0 # 记录loss
                running_corrects = 0 # 记录分类正确数量

                for inputs, labels in tqdm(dataloaders["valid"], desc="valid"):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs) # (B, ClassPred)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / datasetsSize["valid"]
            epoch_acc = running_corrects / datasetsSize["valid"]

            writer.add_scalar('Loss/valid', epoch_loss, epoch)
            writer.add_scalar('Acc/valid', epoch_acc, epoch)
            print(f'Valid Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            if scheduler is not None:
                scheduler.step()

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

def train(model, writer, savePath, dataloaders, device, criterion, optimizer, scheduler=None, num_epochs=25, returnBest=3):
    datasetsSize = {mode: len(dataloaders[mode].dataset)
                    for mode in ['train', 'valid']}

    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir) / Path('best.pth')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0
        iter_count = 0
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
                    iter_loss = loss.item() * inputs.size(0)
                    iter_acc = torch.sum(preds == labels.data).item()
                    running_loss += iter_loss
                    running_corrects += iter_acc

                    writer.add_scalar(f'Iteration Loss/{phase}', iter_loss, iter_count)
                    writer.add_scalar(f'Iteration Acc/{phase}', iter_acc, iter_count)
                    iter_count += 1
                    
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