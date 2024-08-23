from resnet import ResNet
from torchvision.models import resnet18
import torch

def new_model():
    #model = ResNet([2, 2, 2, 2], num_classes=100)
    model = resnet18(num_classes=100)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = torch.nn.Identity()

    return model.cuda()

def reload_model():
    net = new_model()
    net.load_state_dict(torch.load(f'./model_100/292.pth'))
    net.zero_grad()
    return net

'''
def loss_gen(trainloader, criterion):
    loss = 0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = net(inputs)
        loss += criterion(outputs, labels)
    return loss
'''

def evaluatea(net, testloader):
    net.eval()
    with torch.no_grad():
        correct = 0
        correct5 = 0
        total = 0
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs: torch.Tensor = net(images)
            pred = outputs.topk(1, 1)
            pred5 = outputs.topk(5, 1)
            total += labels.size(0)
            correct += (pred[1] == labels.view(-1, 1).expand_as(pred[1])).sum().item()
            correct5 += (pred5[1] == labels.view(-1, 1).expand_as(pred5[1])).sum().item()
        print(f'Accuracy Top 1 of the network on the test images: {100 * correct / total} %')
        print(f'Accuracy Top 5 of the network on the test images: {100 * correct5 / total} %')
    return correct / total, correct5 / total

def fusenet(net):
    net.eval()
    with torch.no_grad():
        fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
        net.conv1 = fuse(
            net.conv1,
            net.bn1
        )

        net.bn1 = torch.nn.Identity()

        def fuselayer(layer):
            layer.conv1 = fuse(layer.conv1, layer.bn1)
            layer.bn1 = torch.nn.Identity()
            layer.conv2 = fuse(layer.conv2, layer.bn2)
            layer.bn2 = torch.nn.Identity()
        
        for layer in sum(([x[0], x[1]] for x in [net.layer1, net.layer2, net.layer3, net.layer4]), []):
            fuselayer(layer)
    return net