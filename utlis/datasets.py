import json
import pickle
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder

def readPickle(dataLoadPath:Path,
               imgPickleFile = Path("dataimg.pickle"),
               lblPickleFile = Path("datalbl.pickle")):
    with open(dataLoadPath / imgPickleFile, "rb") as f:
        data_img = pickle.load(f)
    with open(dataLoadPath / lblPickleFile, "rb") as f:
        data_lbl = pickle.load(f)
    return data_img, data_lbl


def savePickle(imgDict:dict,
               lblDict:dict,
               pickleDataPath:Path,
               imgPickleFile = Path("dataimg.pickle"),
               lblPickleFile = Path("datalbl.pickle")):
    with open(pickleDataPath / imgPickleFile, "wb") as f:
        pickle.dump(imgDict, f, pickle.HIGHEST_PROTOCOL)
    with open(pickleDataPath / lblPickleFile, "wb") as f:
        pickle.dump(lblDict, f, pickle.HIGHEST_PROTOCOL)


def changeParentPath(path, changeParentIndex, changeParentName):
    """改变path中的某个父文件夹"""
    # eg: changeParentPath(Path("./DataSets/ChestXRay2017/train/NORMAL/IM-0115-0001.jpeg), 1, "ChestXRay2017_resize320")
    #     ->Path("./DataSets/ChestXRay2017_resize320/train/NORMAL/IM-0115-0001.jpeg)，第 1 位的父文件夹并修改
    rawPath = path
    parentPathList:list = []
    while 1:
        parentPathList.append(rawPath.name)
        rawPath = rawPath.parent

        if rawPath.name == "":
            break
    newPath = Path("./")
    for i in range(len(parentPathList) - 1):
        index = - i - 1
        if index == - changeParentIndex - 1:
            newPath = newPath / Path(changeParentName)
        else:
            newPath = newPath / Path(parentPathList[index])
    Path.mkdir(newPath, parents=True, exist_ok=True)
    return newPath / Path(parentPathList[0])


def buildChestRay2017(DatasetPath = Path("./DataSets/ChestXRay2017")):
    """建立ChestRay2017数据集的标签字典label.json以及图像路径"""
    imgDict:dict = {}
    lblDict:dict = {}
    DatasetPath = Path.cwd() / DatasetPath

    folders:list = []
    for item in DatasetPath.iterdir():
        if item.is_dir():
            folders.append(DatasetPath / Path(item))

    # 打印所有文件夹的名称
    for folder in folders:
        imgList:list = []
        lblList:list = []

        subFolder:list = []
        for item in folder.iterdir():
            subFolder.append(folder / Path(item))
        
        for patientFolder in subFolder:
            for file in patientFolder.rglob("*.jpeg"):
                imgList.append(Path(file))
                if patientFolder.name == "NORMAL":
                    lblList.append("NORMAL")
                else:
                    split = file.name.split("_")
                    lblList.append(split[1])
        imgDict[folder.name] = imgList
        lblDict[folder.name] = lblList
    classes = set(lblDict["train"] + lblDict["valid"]) # 获得所有标签类型，不重复
    classes = sorted(classes) # 排一下序，保证顺序是一样的
    class_to_index = {}
    for index, class_name in enumerate(classes):
        class_to_index[class_name] = index # 标签转为索引 eg: apple = 1
    with open(DatasetPath / Path("class.json"), "w") as f:
        json.dump(class_to_index, f)
    lblDict["train"] = [class_to_index[class_name] for class_name in lblDict["train"]]
    lblDict["valid"] = [class_to_index[class_name] for class_name in lblDict["valid"]]
    savePickle(imgDict, lblDict, DatasetPath)


class ChestRay2017(Dataset):
    def __init__(self, folder:Path, transform, isTrain = True, jsonFile="class.json"):
        self.folder = folder
        self.transfrom = transform
        self.isTrain = isTrain
        self.image, self.label = self.chosePickle()
        self.class_to_idx, self.classes = self.findClasses(jsonFile=jsonFile)

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]

        imgData = Image.open(image).convert("RGB")
        imgTensor = self.transfrom(imgData)

        return imgTensor, label
    
    def __len__(self):
        return len(self.label)
    
    def chosePickle(self):
        imgDict, lblDict = readPickle(self.folder)
        if self.isTrain:
            return imgDict["train"], lblDict["train"]
        else:
            return imgDict["valid"], lblDict["valid"]
        
    def findClasses(self, jsonFile):
        with open(self.folder / jsonFile, 'r') as f:
            data = json.load(f)
        return data, list(data)

    
def ChestRay2017Binary(folder:Path, transform, isTrain=True):
    phase = "train" if isTrain else "valid"
    return DatasetFolder(root=folder / Path(phase), 
                         loader=lambda file: Image.open(file).convert("RGB"),
                         extensions="jpg", 
                         transform=transform)


if __name__ == "__main__":
    buildChestRay2017(Path("./DataSets/ChestXRay2017_resize320"))
    # DatasetPath = Path("./DataSets/ChestXRay2017")

    # for file in DatasetPath.rglob("*.jpeg"):
    #     image = Image.open(file)
    #     image = image.resize((320, 320), Image.LANCZOS)
    #     savePath = changeParentPath(file, 1, "ChestXRay2017_resize320")
    #     image.save(savePath)
