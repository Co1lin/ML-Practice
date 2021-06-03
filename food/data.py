from pathlib import Path
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class Food(Dataset):

    def __init__(self, dir_path, transform=None, device='cpu', test=False):
        """
        initialize
        """
        self.imgs = []
        self.transform = transform
        self.device = device
        # use pathlib to open images
        dir_path = Path(dir_path)
        img_path_list = list(dir_path.glob('*.jpg'))
        img_path_list.sort()
        for img_file in tqdm(img_path_list):
            img = Image.open(img_file)  # use Image from PIL to open the image
            img_file = img_file.relative_to(dir_path)   # get the filename without dir path
            label = int(str(img_file).split('_')[0]) if test is False else -1
            self.imgs.append((img.copy(), label))
            img.close()

    def __getitem__(self, index):
        """
        :return (PIL_img, label) or (img, label) in torch.Tensor
        """
        if self.transform is not None:
            if type(index) is not slice:
                return ( self.transform(self.imgs[index][0]).to(self.device),
                        torch.LongTensor( [ self.imgs[index][1] ] ).to(self.device) )
            else:
                return [ (self.transform(img).to(self.device), torch.LongTensor([label])).to(self.device)
                        for img, label in self.imgs[index] ]
        else:
            return self.imgs[index]

    def __len__(self):
        return len(self.imgs)

train_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ToTensor(),
    #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    #T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

if __name__ == '__main__':
    dataset = Food('./data/training', train_transform)
    for (img, label) in dataset[:2]:
        print(img, label)