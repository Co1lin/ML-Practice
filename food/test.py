import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter
from data import *
from models import ResNet
from torchvision.models import resnet34, resnet18

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Tester():

    def __init__(self, test_loader, model_path, val_loader):
        """

        :return:
        """
        self.model = torch.load(model_path)
        self.test_loader = test_loader
        self.val_loader = val_loader

    def predict(self, x):
        with torch.no_grad():
            return torch.argmax(self.model(x), dim=1)

    def get_predictions(self):
        self.model.eval()
        preds = torch.LongTensor().to(device)
        for x, _ in tqdm(self.test_loader):
            y = self.predict(x)
            preds = torch.cat([preds, y])
        preds = torch.flatten(preds)
        return preds

    def validate(self):
        self.model.eval()
        labels = torch.LongTensor().to(device)
        preds = torch.LongTensor().to(device)
        for x, label in tqdm(self.val_loader):
            labels = torch.cat([labels, label])
            y = self.predict(x)
            preds = torch.cat([preds, y])
        labels = torch.flatten(labels)
        preds = torch.flatten(preds)
        acc = sum(labels == preds) / labels.shape[0]
        return acc.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Food Image Classification',
        description='Test Script',
        allow_abbrev=True,
    )

    parser.add_argument('-m', '--model', dest='model', type=str, default=None)
    parser.add_argument('-o', '--output-path', dest='out', default=None, type=str)
    args = parser.parse_args()

    test_set = Food('./data/testing', test_transform, device=device, test=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    val_set = Food('./data/validation', test_transform, device=device)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    tester = Tester(test_loader, args.model, val_loader)
    preds = tester.get_predictions()
    print(tester.validate())

    if args.out is None:
        save_path = str(args.model).split("/")[-1].split(".")[-2] + '.csv'
    else:
        save_path = args.out

    with open(save_path, 'w') as f:
        f.write('Id,Category\n')
        for i, pred in enumerate(preds):
            f.write('{},{}\n'.format(i, pred))