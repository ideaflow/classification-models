import cv2
import math
import torch
import torchvision
import numpy as np
from .gradcam import GradCam,show_cam_on_image

ToPILImage = torchvision.transforms.ToPILImage()
ToTensor=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#TODO 这个函数还没写好
def vis_cam(net,images,labels,store_path):

    grad_cam = GradCam(model=net, feature_module=net.layer4, target_layer_names=["2"], use_cuda=True)

    # cam_images=torch.zeros_like(images)
    rows=round(math.sqrt(images.shape[0]))
    columns=math.ceil(images.shape[0]/rows)
    cam_images=np.zeros((rows*images.shape[2],columns*images.shape[3],3),dtype=np.uint8)

    for i in range(images.shape[0]):
        cam = grad_cam(images[i].unsqueeze(0), labels[i].item())
        cv_img = cv2.cvtColor(np.asarray(ToPILImage(images[i] * STD + MEAN)), cv2.COLOR_RGB2BGR)
        cam_img = show_cam_on_image(cv_img, cam)
        cur_x=i%columns*images.shape[2]
        cur_y=i//columns*images.shape[3]
        cam_images[cur_x:cur_x+images.shape[2],cur_y:cur_y+images.shape[3]]=cam_img
    #     cam_images[i] = torchvision.transforms.ToTensor()(Image.fromarray(cv2.cvtColor(cam_img,cv2.COLOR_BGR2RGB)))
    # create grid of images
    # img_grid = torchvision.utils.make_grid(images)
    cv2.imwrite(store_path,cam_images)
    # write to tensorboard
    # writer.add_image('train/class_activate_map', img_grid)
    # TODO 是否需要手动删除对象，释放内存？
    # del grad_cam,cam_images
    # gc.collect()