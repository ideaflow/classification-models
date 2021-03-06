import os
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from Attention import CBAM,SELayer

__all__ = ['ResNet']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

#TODO:BasicBlock暂未修改
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, cbam=None, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if cbam is not None:
            self.cbam = CBAM(planes)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.cbam is not None:
            out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, attn=None, downsample=None,**kwargs):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if attn=='cbam':
            self.attn = CBAM(planes * Bottleneck.expansion)
        elif attn=='se':
            self.attn=SELayer(planes * Bottleneck.expansion,kwargs['reduction'])
        else:
            self.attn=None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.attn is not None:
            out = self.attn(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self,depth,layers,num_classes,pretrained,logger=None,attn=None,fea_norm=False,fc_bias=False,
                 weight_norm=False,**kwargs):
        #depth：网络深度
        #layers：各结构块组中结构块的重复次数
        #num_classes：分类的类别数
        #pretrained：是否加载预训练权重
        #attn：注意力机制类型
        #fea_norm：分类器是否执行特征归一化
        #fc_bias：分类器是否需要偏置项
        #weight_norm：分类器是否执行特征归一化
        #kwargs['pretrained_path']：预训练权重路径
        #kwargs['frozen']：已加载权重的层是否需要冻结
        # kwargs['reduction']：se注意力中bottleneck通道数减少的倍数
        super(ResNet,self).__init__()
        self.logger=logger
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if depth in [50,101,152]:
            block=Bottleneck
        elif depth in [18,34]:
            block=BasicBlock
        else:
            raise NameError('ResNet: 未定义的网络深度: %d'%depth)
        self.layer1 = self._make_layer(block, 64, layers[0], attn,**kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], attn, stride=2,**kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], attn, stride=2,**kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], attn, stride=2,**kwargs)
        self.fea_norm = fea_norm
        self.weight_norm=weight_norm
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=fc_bias)
        # 初始化网络的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if pretrained:
            if 'pretrained_path' in kwargs.keys():
                model_dir,file_name=os.path.split(kwargs['pretrained_path'])
            else:
                model_dir, file_name=None,None
            self.load_state_dict(model_zoo.load_url(model_urls['resnet%d'%depth],model_dir=model_dir,file_name=file_name),
                                load_in_order=kwargs['load_in_order'] if 'load_in_order' in kwargs else False,
                                frozen=kwargs['frozen'])

    def _make_layer(self, block, planes, blocks, attn=None, stride=1,**kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []

        layers.append(block(self.inplanes, planes, stride=stride, attn=attn, downsample=downsample,**kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, attn=attn,**kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.spp(x)
        if self.fea_norm:
            x=F.normalize(x,dim=-1)
            x=self.fc(x*100.)
        else:
            x = self.fc(x)

        return x

    def get_features(self):
        # 仅提取特征，不分类
        return nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        )

    def load_state_dict(self, state_dict, load_in_order=False, frozen=False):
        model_dict = self.state_dict()
        pretrained_dict = {}
        if load_in_order:
            for (km,vm),(ks,vs) in zip(model_dict.items(),state_dict.items()):
                if vm.size()==vs.size():
                    pretrained_dict[km]=vs
        else:
            for k, v in state_dict.items():
                k=k.replace('module.','')
                k=k.replace('cbam.','attn.')
                if k in model_dict and model_dict[k].size() == v.size():
                    pretrained_dict[k]=v

        if frozen:
            for p in self.named_parameters():
                if p[0] in pretrained_dict:
                    p[1].requires_grad = False
        not_loaded_keys = [k for k in model_dict.keys() if k not in pretrained_dict.keys()]
        if len(not_loaded_keys) == 0:
            res_str='%s: All params loaded' % type(self).__name__
        else:

            res_str='%s: Some params were not loaded:' % type(self).__name__+('\n%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys)
        if self.logger: self.logger.info(res_str)
        print(res_str)
        model_dict.update(pretrained_dict)
        super(ResNet, self).load_state_dict(model_dict)
