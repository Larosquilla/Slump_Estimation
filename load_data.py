from config import Config
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class MyDataset(Dataset):
    def __init__(self, root_dir, label):
        self.img_dir = os.path.join(root_dir, label)
        self.img_list = os.listdir(self.img_dir)
        self.data_transformers = transforms.Compose([transforms.Resize((Config.get_input_h(), Config.get_input_w())),
                                                     transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_path)
        img = img.convert("RGB")
        input = self.data_transformers(img)
        target = Config.get_class_label().index(float(os.path.basename(self.img_dir)))
        return input, target

    def __len__(self):
        return len(self.img_list)


def make_dataset(root_dir):
    temp = []
    labels = os.listdir(root_dir)
    for label in labels:
        temp.append(MyDataset(root_dir, label))
    sum = temp[0]
    for i in range(1, len(temp)):
        sum = sum + temp[i]
    return sum


def make_dataLoader(dataset):
    dataLoader = DataLoader(dataset=dataset, batch_size=Config.get_batch_size(), shuffle=True, num_workers=0,
                            drop_last=False)
    return dataLoader
