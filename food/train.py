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


class Trainer():

    def __init__(self, train_loader, val_loader,
                 model: nn.Module, optimizer: optim.Optimizer,
                 loss_fn: nn.Module, save_path: str = './ckpts/',
                 writer: SummaryWriter = None,
                 total_epoch=1000):
        """

        :param train_loader:
        :param val_loader:
        :param model:
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.total_epoch = total_epoch
        self.loss_fn = loss_fn
        self.writer = writer
        self.save_path = save_path
        if self.writer is not None:
            # draw the model
            self.writer.add_graph(model, (torch.zeros(1, 3, 224, 224).to(device)))

    def predict(self, x):
        with torch.no_grad():
            return torch.argmax(self.model(x), dim=1)

    def step(self, x, label):
        self.model.train()
        self.optimizer.zero_grad()
        y = self.model(x)
        label = torch.flatten(label)
        loss = self.loss_fn(y, label)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.model.train()

        tot_steps = 0
        max_val_acc = 0

        for epoch in range(self.total_epoch):
            # validate before starting an epoch
            val_acc = self.validate()
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                torch.save(self.model, '{}epoch_{}_{:.3f}.pkl'.format(self.save_path, epoch, val_acc))
            if writer is not None:
                self.writer.add_scalar('Val/acc', val_acc, epoch)

            with tqdm(total=len(self.train_loader)) as t:
                t.set_description(f"Epoch: {epoch}")
                for x, label in self.train_loader:
                    tot_steps += 1
                    loss = self.step(x, label)
                    t.set_postfix(loss='{:.5f}'.format(loss), last_val_acc='{:.2f}'.format(val_acc))
                    t.update(1)
                    if self.writer is not None:
                        self.writer.add_scalar('Train/loss', loss, tot_steps)
            # finish an epoch

    def validate(self):
        self.model.eval()
        labels = torch.LongTensor().to(device)
        preds = torch.LongTensor().to(device)
        for x, label in self.val_loader:
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
        description='Train Script',
        allow_abbrev=True,
    )

    parser.add_argument('-m', '--model', dest='model', type=str, default='myres', help="name of the model to use")
    parser.add_argument('-pre', '--pre_trained', dest='pre', action='store_true')
    parser.add_argument('-dev', '--device', dest='device', type=str, default='cuda:0')
    parser.add_argument('-opt', '--optimizer', dest='optim', type=str, default='adam')
    args = parser.parse_args()
    print(f'pretrained: {args.pre}')

    if device == 'cuda':
        device = args.device

    # get dataset on device
    train_set = Food('./data/training', train_transform, device=device)
    val_set = Food('./data/validation', test_transform, device=device)
    # get dataloader
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    num_classes = 11
    model_name = args.model
    if model_name == 'myres':
        model = ResNet(num_classes=num_classes, repeats=[0, 1, 1, 1, 1]).to(device)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=args.pre)
        fc_in = model.fc.in_features
        model.fc = nn.Linear(fc_in, num_classes)
        model = model.to(device)
    elif model_name == 'resnet18':
        model = resnet18(pretrained=args.pre)
        fc_in = model.fc.in_features
        model.fc = nn.Linear(fc_in, num_classes)
        model = model.to(device)
    else:
        raise ValueError(f'wrong model name: {args.model}')

    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    elif args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

    writer = SummaryWriter(f'log_{model_name}')
    save_path = f'./ckpts_{model_name}_{args.optim}/'
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    trainer = Trainer(train_loader, val_loader, model, optimizer, nn.CrossEntropyLoss(), writer=writer, save_path=save_path)
    trainer.train()