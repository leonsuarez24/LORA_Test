from tqdm import tqdm
import torch
from networks import DemoNet
from torchmetrics.classification import MulticlassAccuracy
from torchvision.datasets import MNIST
from torchvision import transforms
import numpy as np
from torchsummary import summary
from utils import AverageMeter


def train_demo():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = DemoNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    ACCURACY = MulticlassAccuracy(num_classes=10).to(device)

    net.to(device)

    print(summary(net, (1, 28, 28)))

    dst_train = MNIST('data', train=True, download=True, transform=transforms.ToTensor())  
    dst_test = MNIST('data', train=False, download=True, transform=transforms.ToTensor())

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=32, shuffle=False)

    for epoch in range(5):
        net.train()
        data_loop_train = tqdm(enumerate(trainloader), total=len(trainloader), colour='red')
        train_accuracy = AverageMeter()
        train_loss = AverageMeter()
        for _, data in data_loop_train:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            accuracy = ACCURACY(outputs, labels)

            train_accuracy.update(accuracy.item(), inputs.size(0))
            train_loss.update(loss.item(), inputs.size(0))
            data_loop_train.set_description(f'Epoch {epoch+1}/{10}')
            data_loop_train.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)

        with torch.no_grad():
            net.eval()
            data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour='green')
            test_accuracy = AverageMeter()
            test_loss = AverageMeter()
            for _, data in data_loop_test:
                inputs, labels = data
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = net(inputs)
                accuracy = ACCURACY(outputs, labels)
                loss = criterion(outputs, labels)
                test_loss.update(loss.item(), inputs.size(0))
                test_accuracy.update(accuracy.item(), inputs.size(0))
                data_loop_test.set_description(f'Epoch {epoch+1}/{10}')
                data_loop_test.set_postfix(loss=test_loss.avg, accuracy=accuracy.item())

    torch.save(net.state_dict(), 'model_MNIST.pth')

if __name__ == '__main__':
    train_demo()