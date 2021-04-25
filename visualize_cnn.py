"""EVALUATION
Created: Nov 22,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
# import matplotlib.pyplot as plt
import argparse
import models
from gradcam import GradCam,show_cam_on_image
from utils import get_transform
from PIL import Image
# GPU settings
assert torch.cuda.is_available()

device = torch.device("cuda:2")
torch.backends.cudnn.benchmark = True

ToPILImage = transforms.ToPILImage(mode='F')
# MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
# STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
eval_transform=get_transform((448,448),'eval')

def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)

parser = argparse.ArgumentParser(description='训练参数')
parser.add_argument('dataset_tag', type=str,choices=['bird','aircraft','dog','car'],help='')
parser.add_argument('net_arch',type=str,choices=['resnet50','resnet50_cbam','resnet50_se','resnet50_coord',
                                                   ],help='网络结构')
parser.add_argument('--norm',action='store_true',help='fea norm')

args = parser.parse_args()

def main():
    ##################################
    # Initialize model
    ##################################
    num_classes={'bird':200,'car':196,'aircraft':100}
    if args.net_arch=='resnet50':
        net = models.resnet50(False, num_classes[args.dataset_tag],fea_norm=args.norm)
    else:
        raise NameError('未实现的网络结构')
    # net = torch.hub.load('/data2/Bruce/Projects/FcaNet', 'fca50', source='local', pretrained=False)
    # net.fc = torch.nn.Linear(2048, 200)

    weight_dir=os.path.join('./FGVC/logs',args.net_arch if not args.norm else args.net_arch+'_norm', args.dataset_tag)
    # Load ckpt and get state_dict
    checkpoint = torch.load(os.path.join(weight_dir,'model.ckpt'))
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)

    ##################################
    # use cuda
    ##################################
    net.to(device)
    # if torch.cuda.device_count() > 1:
    #     net = nn.DataParallel(net)
    net.eval()
    cam_extractor = GradCam(model=net, feature_module=net.layer4, target_layer_names=["2"], use_cuda=True)
    # cam_extractor = GradCam(model=net, feature_module=net[7], target_layer_names=["2"], use_cuda=True)
    while True:
        src=input('image path:')
        if src == 'q':
            break
        if not os.path.isfile(src):
            print('invalid path')
            continue
        _,filename=os.path.split(src)
        real_filename,ext=os.path.splitext(filename)
        pil_img=Image.open(src).convert('RGB')
        cv_img=cv2.cvtColor(np.asarray(pil_img),cv2.COLOR_RGB2BGR)
        # 将tensor转换为pil图片显示>
        # plt.figure()
        # plt.imshow(pil_img)
        # plt.show()

        img_tensor=eval_transform(pil_img).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        # Retrieve the CAM
        grayscale_cam = cam_extractor(img_tensor, None)
        grayscale_cam = cv2.resize(grayscale_cam, (pil_img.size[0], pil_img.size[1]))
        cam,heatmap,gray_heatmap = show_cam_on_image(cv_img, grayscale_cam)
        cv2.imwrite(os.path.join(weight_dir,real_filename+'_cam'+ext),cam)
        cv2.imwrite(os.path.join(weight_dir,real_filename+'_heatmap'+ext),heatmap)
        cv2.imwrite(os.path.join(weight_dir,real_filename+'_graymap'+ext),gray_heatmap)
        # cv2.namedWindow('watch',0)
        # cv2.imshow('watch',cam)
        # cv2.waitKey(0)
        # cv2.destroyWindow('watch')
        # plt.figure()
        # plt.imshow(result)
        # plt.show()


if __name__ == '__main__':
    main()
