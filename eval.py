import torch
from typing import Tuple
from utlis import *
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.models import resnet18, resnet50
from torchvision.transforms import v2 as transforms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


@torch.no_grad()
def get_all_preds(model: torch.nn.Module,
                  loader: torch.utils.data.DataLoader) -> torch.Tensor:
    '''获取该model对该loader所有类别的预测概率'''
    all_preds = torch.Tensor([]).cuda()
    for inputs, _ in loader:
        inputs = inputs.cuda()
        batch_preds = model(inputs)
        all_preds = torch.cat((all_preds, batch_preds), dim=0)
    return all_preds

def get_num_correct(preds:torch.Tensor, 
                    labels: list, 
                    n:int = 1) -> Tuple[float, float]:
    '''获得top-n的预测准确量'''
    labels:torch.Tensor = torch.Tensor(labels).cuda()
    _, idxs = preds.topk(n, dim=1, largest=True, sorted=True)
    # NUM -> [NUM, 1] -> [NUM, n]
    labels = labels.view(labels.numel(), -1).expand_as(idxs)

    correct:torch.Tensor = idxs.eq(labels).float()
    correct_n = correct[:, :2].sum() # 取前两列，注意索引是[首:尾]，尾不计入
    correct_1 = correct[:, :1].sum() # 取第一列
    return correct_1.item(), correct_n.item()

if __name__ == "__main__":
    model = resnet18(weights=None)
    model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, 3))
    model_state_dict = torch.load(Path("./runs/ChestXRay2017_resize320/resnet18 0.8878/best.pth"))["model_state"]  # 读取模型权重
    model.load_state_dict(model_state_dict, strict=True)  # 严格匹配模型权重到模型中
    
    model = model.cuda().eval()

    transform = transforms.Compose([transforms.ToImage(),
                                    transforms.ToDtype(torch.float32, scale=True),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    
    datasetPath = Path("./DataSets/ChestXRay2017_resize320")
    dataset = ChestRay2017(datasetPath, transform, isTrain=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = get_all_preds(model, dataloader)
    
    preds_correct, top_n_preds_correct = get_num_correct(all_preds, dataset.label, 2)
    print('total correct:', preds_correct, top_n_preds_correct)
    print('accuracy:', preds_correct / len(dataset), top_n_preds_correct/ len(dataset))

    cm = confusion_matrix(dataset.label, all_preds.argmax(dim=1).tolist())
    print(cm)
    plt.figure(figsize=(8,7))
    plot_confusion_matrix(cm, dataset.classes)
    # plt.show()
    plt.savefig("./cm.jpg")
    log_train = {}

    all_idx_preds = all_preds.argmax(dim=1).tolist()
    log_train['Valid-Accuracy'] = accuracy_score(dataset.label, all_idx_preds)
    log_train['Valid-Precision'] = precision_score(dataset.label, all_idx_preds, average='macro')
    log_train['Valid-Recall'] = recall_score(dataset.label, all_idx_preds, average='macro')
    log_train['Valid-F1-Score'] = f1_score(dataset.label, all_idx_preds, average='macro')

    print(classification_report(dataset.label, all_idx_preds, target_names=dataset.classes))
    print(log_train)
