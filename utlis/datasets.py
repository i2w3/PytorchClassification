import json
import shutil
import random
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


def splitData(inputPath: Path, 
              outputPath: Path, 
              rate: float = 0.2, 
              extensions: list = ["jpg", "jpeg", "png"]):
    random.seed(0)
    # 创建文件夹
    Path.mkdir(outputPath, exist_ok=True)
    shutil.rmtree(outputPath)
    train_image_path = outputPath / Path("train")
    valid_image_path = outputPath/ Path("valid")
    Path.mkdir(train_image_path, parents=True, exist_ok=True)
    Path.mkdir(valid_image_path, parents=True, exist_ok=True)

    imagesList: list = []
    for suffix in extensions:
        for file in inputPath.rglob("*." + suffix):
            imagesList.append(file)

    num = len(imagesList)
    print(f"{inputPath}下检测到{num}张图片...")
    valid_index = random.sample(imagesList, k=int(num * rate))
    for single_image in imagesList:
        if single_image in valid_index:
            shutil.copy(single_image, valid_image_path)
        else:
            shutil.copy(single_image, train_image_path)
    print("数据集划分完成")

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
    """建立ChestRay2017数据集的标签字典class.json以及图像路径"""
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


def buildNEC(DatasetPath = Path("./DataSets/NEC")):
    # 0、清空
    for phase in ["train", "valid"]:
        Path.mkdir(DatasetPath / Path(phase), exist_ok=True)
        shutil.rmtree(DatasetPath / Path(phase))
    # 1、划分NONEC
    outputPath = DatasetPath / Path("NOTNEC")
    splitData(DatasetPath / Path("raw/对照"), outputPath)
    for phase in ["train", "valid"]:
        folder_path = outputPath / Path(phase)
        savePath = DatasetPath / Path(phase) / Path("NOTNEC")
        Path.mkdir(savePath, parents=True)
        for suffix in ["png", "jpg"]:
            for file in folder_path.rglob("*." + suffix):
                shutil.move(file, savePath)
    shutil.rmtree(outputPath)
    # 2、划分NEC
    outputPath = DatasetPath / Path("NEC")
    splitData(DatasetPath / Path("raw/典型"), outputPath)
    for phase in ["train", "valid"]:
        folder_path = outputPath / Path(phase)
        savePath = DatasetPath / Path(phase) / Path("NEC")
        Path.mkdir(savePath, parents=True)
        for suffix in ["png", "jpg"]:
            for file in folder_path.rglob("*." + suffix):
                shutil.move(file, savePath)
    shutil.rmtree(outputPath)


class ChestRay2017(Dataset):
    def __init__(self, folder:Path, transform, isTrain = True, jsonFile="class.json"):
        self.folder = folder
        self.transfrom = transform
        self.isTrain = isTrain
        self.image, self.label = self.chosePickle()
        self.class_to_idx, self.classes = self.findClasses(jsonFile=jsonFile)
        self.idx_to_class = {y:x for x, y in self.class_to_idx.items()}
        self.statistics = self.statisticsClasses()

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
    
    def statisticsClasses(self):
        idx_count = {}
        for item in self.label:
            if item in idx_count:
                idx_count[item] += 1
            else:
                idx_count[item] = 1
        class_count = {}
        for idx in idx_count:
            class_count[self.classes[idx]] = idx_count[idx]
        return class_count


class ChestRay2017Binary(DatasetFolder):
    def __init__(self, folder: Path, transform, isTrain=True):
        phase = "train" if isTrain else "valid"
        super().__init__(root=folder / Path(phase),
                         loader=lambda file: Image.open(file).convert("RGB"),
                         extensions="jpeg", 
                         transform=transform)
        self.statistics = self.statisticsClasses()
        self.idx_to_class = {y:x for x, y in self.class_to_idx.items()}
        

    def statisticsClasses(self):
        idx_count = {}
        for item in self.targets:
            if item in idx_count:
                idx_count[item] += 1
            else:
                idx_count[item] = 1
        class_count = {}
        for idx in idx_count:
            class_count[self.classes[idx]] = idx_count[idx]
        return class_count
    

class NECBinary(DatasetFolder):
    def __init__(self, folder: Path, transform, isTrain=True):
        phase = "train" if isTrain else "valid"
        self.folder = folder
        super().__init__(root=folder / Path(phase),
                         loader=lambda file: Image.open(file).convert("RGB"),
                         extensions=("png", "jpg"), 
                         transform=transform)
        self.statistics = self.statisticsClasses()
        self.idx_to_class = {y:x for x, y in self.class_to_idx.items()}
        self.buildClassJson()
        
    def statisticsClasses(self):
        idx_count = {}
        for item in self.targets:
            if item in idx_count:
                idx_count[item] += 1
            else:
                idx_count[item] = 1
        class_count = {}
        for idx in idx_count:
            class_count[self.classes[idx]] = idx_count[idx]
        return class_count

    def buildClassJson(self):
        class_to_index = {}
        for index, class_name in enumerate(self.classes):
            class_to_index[class_name] = index # 标签转为索引 eg: apple = 1
        with open(self.folder / Path("class.json"), "w") as f:
            json.dump(class_to_index, f)


if __name__ == "__main__":
    # DatasetPath = Path("./DataSets/ChestXRay2017_resize320")

    # 1. 检索数据集内容，建立pickle和class.json
    buildChestRay2017(Path("./DataSets/ChestXRay2017_resize320"))

    # 2. resize图片大小
    # for file in DatasetPath.rglob("*.jpeg"):
    #     image = Image.open(file)
    #     image = image.resize((320, 320), Image.LANCZOS)
    #     savePath = changeParentPath(file, 1, "ChestXRay2017_resize320")
    #     image.save(savePath)

    
    # buildNEC()
    # DatasetPath = Path("./DataSets/NEC")
    # for phase in ["train", "valid"]:
    #     folder_path = DatasetPath / Path(phase)
    #     for suffix in ["png", "jpg"]:
    #         for file in folder_path.rglob("*." + suffix):
    #             image = Image.open(file)
    #             image = image.resize((320, 320), Image.Resampling.LANCZOS)
    #             savePath = changeParentPath(file, 1, "NEC_resize320")
    #             image.save(savePath)