import torch.nn as nn
import torch.nn.functional as F
import torch


# From DAS article
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, cnn_cfg):
        super(CNN, self).__init__()

        self.k = 1
        #self.flattening = flattening

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [2, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            elif m == 'MH':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x, ))
                    in_channels = x
                    cnt += 1

    def forward(self, x, reduce='max'):

        y = x
        for i, nn_module in enumerate(self.features):
            y = nn_module(y)

        #if self.flattening == 'maxpool':
        y = F.max_pool2d(y, [y.size(2), self.k], stride=[y.size(2), 1], padding=[0, self.k // 2])
        # elif self.flattening == 'concat':
        #     y = y.view(y.size(0), -1, 1, y.size(3))

        return y


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)


class CTCHead(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses):
        super(CTCHead, self).__init__()

        hidden, num_layers = rnn_cfg

        self.rec = nn.LSTM(input_size, hidden, num_layers=num_layers, bidirectional=True, dropout=.2)
        self.fnl = nn.Sequential(nn.Dropout(.2), nn.Linear(2 * hidden, nclasses))

        #self.att = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses), nn.Sigmoid())

        self.cnn = nn.Conv2d(input_size, nclasses, kernel_size=(1, 3), stride=1, padding=(0, 1))

    def forward(self, x):

        # x dimension: batch size, 256, 1,  nb frames
        y = x.permute(2, 3, 0, 1)[0]  # Output dimension: nb frames, batch size, 256
        y = self.rec(y)[0]
        # a = self.att(y)
        # y = (a * self.fnl(y) + (1 - a) * self.cnn(x))
        y = self.fnl(y)  # Output dimension: nb frames, batch size, alphabet size

        y_aux = self.cnn(x).permute(2, 3, 0, 1)[0]

        return y, y_aux


class CRNN(nn.Module):
    def __init__(self, cnn_cfg, head_cfg, nclasses):
        super(CRNN, self).__init__()

        self.features = CNN(cnn_cfg)

        # if flattening == 'maxpool':
        hidden = cnn_cfg[-1][-1]
        # elif flattening == 'concat':
        #     hidden = 2 * 8 * cnn_cfg[-1][-1]
        # else:
        #     print('problem!')

        self.top = CTCHead(hidden, head_cfg, nclasses)

    def forward(self, x):

        y = self.features(x)  # Output dimension: batch size, 256, 1,  nb frames
        y = self.top(y)

        return y

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

# def set_bn_eval(m):
#     if isinstance(m, nn.modules.batchnorm._BatchNorm):
#         m.eval()
