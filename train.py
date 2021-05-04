"""训练主代码
Created: April 23,2021 - wechange
"""
import os
import time
import logging
import warnings
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.hub
# import torchvision
from Visualization import vis_cam

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from DataSampler import get_trainval_datasets
from Backbone import *
from Tools import *
from multiprocessing import cpu_count

# GPU settings
assert torch.cuda.is_available()
device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='训练参数')
dataset_group=parser.add_argument_group('dataset','数据集相关的参数')
dataset_group.add_argument('--dataset_path', metavar='DIR', required=True, help='数据集路径')
dataset_group.add_argument('--dataset_tag', type=str,choices=['bird','aircraft','dog','car'],required=True,help='数据集类型')
dataset_group.add_argument('--image_size',nargs=2, type=int, default=[224,224],help='输入图片的尺寸')

scheduler_group=parser.add_argument_group('scheduler','学习率参数')
scheduler_group.add_argument('--init_lr', type=float, default=0.01, help='初始学习率')
scheduler_group.add_argument('--optim',type=str,choices=['sgd','adam'],default='sgd',help='梯度下降优化算法')
scheduler_group.add_argument('--scheduler',type=str,choices=['step','cosine','none'],default='step',help='梯度更新策略')

performance_group=parser.add_argument_group('performance','硬件相关的参数')
performance_group.add_argument('--workers',type=int,default=8,help='加载数据的进程数')
performance_group.add_argument('--cpu_num',type=int,default=0,help='设置程序使用的cpu核心数，其中有两个特殊值：'
                                                                   '0代表保持pytorch默认的核心数，'
                                                                   '-1代表使用主机的最大核心数')

train_group=parser.add_argument_group('train','网络训练相关的参数')
train_group.add_argument('--epoch',type=int,default=100,help='总共训练几轮')
train_group.add_argument('--train_batch_size', type=int, default=32,help='训练时的批量大小')

test_group=parser.add_argument_group('test','网络测试相关的参数')
test_group.add_argument('--test_batch_size',type=int,default=32,help='总共训练几轮')

network_group=parser.add_argument_group('network','网络结构相关的参数')
network_group.add_argument('--net_arch',type=str,choices=['resnet50','resnet50_cbam','resnet50_se',
                            'resnet50_coord'], required=True,help='网络结构')
network_group.add_argument('--fea_norm',action='store_true',help='是否执行fc层特征归一化')
network_group.add_argument('--weight_norm',action='store_true',help='是否执行fc层权重归一化')
network_group.add_argument('--fc_bias',action='store_true',help='fc层是否需要bias')

pretrain_group=parser.add_argument_group('pretrain','网络预训练权重相关的参数')
pretrain_group.add_argument('--pretrained',action='store_true',help='是否加载imagenet上的预训练权重')
pretrain_group.add_argument('--pretrained_path',metavar='DIR',default=None,help='预训练权重的路径，'
                                '默认加载pytorch官方提供的imagenet预训练权重')
pretrain_group.add_argument('--frozen',action='store_true',help='已加载权重的层是否需要冻结')
pretrain_group.add_argument('--checkpoint_path',metavar='DIR',default='',help='已训练的权重保存位置')

visualize_group=parser.add_argument_group('visualize','网络可视化相关的参数')
visualize_group.add_argument('--visualize',type=str,choices=['','cam','gradcam'],
                             default='',help='使用何种网络可视化方法')

save_group=parser.add_argument_group('save','结果保存相关的参数')
save_group.add_argument('--save_dir',metavar='DIR', required=True,help='训练结果保存路径')
save_group.add_argument('--log_name',type=str,default='train.log',help='日志文件名')
save_group.add_argument('--model_name',type=str,default='model.ckpt',help='权重文件名')
save_group.add_argument('--rm_log',action='store_true',help='训练前是否清空保存目录')
save_group.add_argument('--tensorboard',action='store_true',help='是否使用tensorboard')

loss_group=parser.add_argument_group('loss','损失函数相关的参数')
loss_group.add_argument('--loss_type',type=str,choices=['softmax_ce','center_loss','arcface_loss'],default='softmax_ce',
                        help='损失函数类型')
loss_group.add_argument('--margin',type=float,default=0.0,help='损失中的间隔大小')

args = parser.parse_args()

# Loss functions
if args.loss_type=='softmax_ce':
    loss = nn.CrossEntropyLoss()
else:
    raise NameError('未定义的损失类型')

# Loss and metric
loss_container = AverageMeter(name='Loss')
raw_metric = TopKAccuracyMetric(topk=(1, 5))
# center_loss = CenterLoss()
# circle_loss_=nn.Softplus()

#设置可以使用的cpu核心数
if args.cpu_num:
    if args.cpu_num>0:
        cpu_num = 1  # cpu_count() # 自动获取最大核心数目
    elif args.cpu_num==-1:
        cpu_num=cpu_count()
    else:
        raise NameError('不合法的cpu核心数：%d'%args.cpu_num)

    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

# ToPILImage = torchvision.transforms.ToPILImage()
# ToTensor=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
# STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

if args.rm_log and os.path.isdir(args.save_dir):
    shutil.rmtree(args.save_dir)
os.makedirs(args.save_dir,exist_ok=True)
if args.tensorboard:
    writer = SummaryWriter(args.save_dir)

images,labels=None,None

def main():
    global images,labels
    ##################################
    # Logging setting
    ##################################
    logging.basicConfig(
        filename=os.path.join(args.save_dir, args.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Load dataset
    ##################################

    train_dataset, validate_dataset = get_trainval_datasets(args.dataset_tag, args.image_size,args.dataset_path)

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=args.test_batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
    num_classes = train_dataset.num_classes
    ##################################
    # Initialize model
    ##################################
    if args.visualize:
        images,labels=next(iter(validate_loader))

    if args.net_arch=='resnet50':
        net=ResNet(50,[3,4,6,3],num_classes,args.pretrained,fea_norm=args.fea_norm,fc_bias=args.fc_bias,
                   weight_norm=args.weight_norm,frozen=args.frozen,pretrained_path=args.pretrained_path)
    else:
        raise NameError('未实现的网络结构')

    logs = {}
    start_epoch = 0
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict,frozen=args.frozen)
    # feature_center: size of (#classes, #attention_maps * #channel_features)
    # feature_center = torch.zeros(num_classes, net.num_features).to(device)

    ##################################
    # Optimizer, LR Scheduler
    ##################################
    learning_rate = logs['lr'] if 'lr' in logs else args.init_lr
    if args.optim=='sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optim=='adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate,amsgrad=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    scheduler=None
    if args.scheduler=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    elif args.scheduler=='cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * (args.epoch - start_epoch),
                                                           eta_min=0, last_epoch=-1, verbose=True)

    ##################################
    # Use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        # net = nn.DataParallel(net,device_ids=[2,3])
        net = nn.DataParallel(net)

    logging.info('Network weights save to {}'.format(args.save_dir))

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * (config.epochs - start_epoch),
    #                                                        eta_min=0, last_epoch=-1, verbose=True)

    ##################################
    # ModelCheckpoint
    ##################################
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(args.save_dir, args.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    ##################################
    # TRAINING
    ##################################
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(args.epoch, args.train_batch_size, len(train_dataset), len(validate_dataset)))
    logging.info('')

    for epoch in range(start_epoch, args.epoch):
        callback.on_epoch_begin()

        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        train(logs=logs,
              data_loader=train_loader,
              net=net,
              # feature_center=feature_center,
              optimizer=optimizer)
        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 batch_index=epoch*len(train_loader))
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(logs['val_loss'])
            else:
                scheduler.step()

        callback.on_epoch_end(logs, net)#, feature_center=feature_center)
        if args.tensorboard:
            writer.flush()

def train(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    # feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    epoch=logs['epoch']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    start_time = time.time()
    net.train()
    with tqdm(total=len(data_loader), unit=' batches',ncols=0) as pbar:
        for i, (X, y) in enumerate(data_loader):
            pbar.set_description(
                'Epoch {}/{}, Learning Rate {:g}, Progress Rate'.format(epoch, args.epoch, optimizer.param_groups[0]['lr']))

            optimizer.zero_grad()

            # obtain data for training
            X = X.to(device)
            y = y.to(device)
            # blur_img=blur_img.to(device)

            ##################################
            # Raw Image
            ##################################
            # raw images forward
            # y_pred_raw,y_pred_parts,y_pred_drop= net(X)

            y_pred = net(X)
            # Update Feature Center
            # feature_center_batch = F.normalize(feature_center[y], dim=-1)
            # feature_center[y] += config.beta * (raw_feature.detach() - feature_center_batch)

            ##################################
            # Attention Cropping
            ##################################
            # with torch.no_grad():
            #     crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

            # crop images forward
            # y_pred_crop, _, _ = net(crop_images)

            ##################################
            # Attention Dropping
            ##################################
            # with torch.no_grad():
            #     drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

            # drop images forward
            # y_pred_drop, _, _ = net(drop_images)

            # Loss
            # batch_loss = circle_loss_(raw_loss).mean() + \
            batch_loss = loss(y_pred, y)
            # center_loss(raw_feature, feature_center_batch)

            # backward
            batch_loss.backward()
            optimizer.step()

            # metrics: Loss and top-1,5 error
            with torch.no_grad():
                epoch_loss = loss_container(batch_loss.item())
                epoch_acc = raw_metric(y_pred, y)
            pbar.set_postfix({'Train Loss': '{:.5f}'.format(epoch_loss),
                              'Train Acc': '{:.2f}%, {:.2f}%'.format(epoch_acc[0], epoch_acc[1])})
            # end of this batch
            batch_info = 'Train Loss {:.4f}, Raw Acc ({:.2f}, {:.2f})'.format(
                epoch_loss, epoch_acc[0], epoch_acc[1])

            # batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f})'.format(
            #     epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],epoch_drop_acc[0],epoch_drop_acc[1])
            # train_acc_writer.add_scalar('acc',epoch_raw_acc[0],logs['epoch']*len(data_loader)+i)
            # pbar.set_postfix_str(batch_info)
            pbar.update()
            # print(batch_info, end='')

    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_acc
    if args.tensorboard:
        writer.add_scalar('Loss/train', epoch_loss, (logs['epoch'] - 1))
        writer.add_scalar('Accuracy/train', epoch_acc[0], (logs['epoch'] - 1))

    # begin training
    logs['train_info'] = batch_info
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))


def validate(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    epoch=logs['epoch']
    # batch_index=kwargs['batch_index']
    # pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    net.eval()
    with tqdm(total=len(data_loader), unit=' batches',ncols=0) as pbar:
        with torch.no_grad():
            for i, (X, y) in enumerate(data_loader):
                pbar.set_description('Epoch {}/{}, Process Rate'.format(epoch, args.epoch))

                # obtain data
                X = X.to(device)
                y = y.to(device)

                ##################################
                # Raw Image
                ##################################
                y_pred = net(X)

                ##################################
                # Object Localization and Refinement
                ##################################
                # crop_images = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
                # y_pred_crop, _, _ = net(crop_images)

                ##################################
                # Final prediction
                ##################################
                # y_pred = (y_pred_raw + y_pred_crop) / 2.

                # Loss
                batch_loss = loss(y_pred, y)
                # batch_loss = circle_loss_(raw_loss).mean()
                epoch_loss = loss_container(batch_loss.item())

                # metrics: top-1,5 error
                epoch_acc = raw_metric(y_pred, y)

                pbar.set_postfix({'Val Loss': '{:.5f}'.format(epoch_loss),
                                  'Val Acc': '{:.2f}%, {:.2f}%'.format(epoch_acc[0], epoch_acc[1])})
                pbar.update()
    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    if args.visualize:
        vis_cam(net,images,labels,os.path.join(args.save_dir,'cam_epoch%d.jpg'%epoch))
    end_time = time.time()
    if args.tensorboard:
        writer.add_scalar('Loss/test', epoch_loss, logs['epoch']-1)
        writer.add_scalar('Accuracy/test', epoch_acc[0], logs['epoch']-1)
    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    # test_loss_writer.add_scalar('Loss', epoch_loss, batch_loss)
    # test_acc_writer.add_scalar('acc',epoch_acc[0],batch_index)
    # pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))
    # print(batch_info)
    # write log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    logging.info('')


if __name__ == '__main__':
    main()
    if args.tensorboard:
        writer.close()
