# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:51:33 2021

@author: malrawi

"""

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from googlenet import googlenet

from PIL import Image

# from audtorch.metrics.functional import pearsonr
from CKC_net import CKCNet
import numpy as np
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet # Install with: pip install efficientnet_pytorch 
''' https://github.com/lukemelas/EfficientNet-PyTorch '''

from datetime import datetime
# from PIL import Image

import sys
import platform

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'),  help='use pre-trained model')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument("--redirect-std-to-file", default=False, type=lambda x: (str(x).lower() == 'true'),  help="True/False - default False; if True sets all console output to file")
parser.add_argument("--experiment-name", type=str, default='results', help="name of the folder inside saved_models")




# parser.add_argument("--local_rank", type=int, default=0)


# HOW TO SET MULTI PROCESSING  
#  https://github.com/NVIDIA/apex/blob/master/examples/simple/distributed/distributed_data_parallel.py

best_acc1 = 0
error_metric = nn.MSELoss(reduction='mean')


def main():
    args = parser.parse_args()    
    
    if platform.system() == 'Windows': 
        args.workers= 0 # change to 0 if there's a problem
    # used for debugging    
    # args.batch_size = 16
    # args.print_freq = 200
    
    # args.redirect_std_to_file = False
    args.min_kohonen_dim = 500
    args.pretrained=True
    args.dataset_name = 'cifar100'
    args.arch = 'googlenet' # 'CnnKohonen' # 'effnet' 'resnet18'
    # args.kohonen_dim=1000
    # args.neighbor_radius=100
    # args.epochs=200
    if args.arch =='googlenet':
        args.pretrained=False
    
        
    if  args.redirect_std_to_file:    # sending output to file                
        out_file_name = (args.dataset_name +'-'+args.arch+'-' +
                         str(datetime.now().ctime())).replace(':','-')
        print('Output sent to ', out_file_name)
        sys.stdout = open(out_file_name+'.txt',  'w')
    
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print('No GPUs per node', ngpus_per_node)
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def get_data(dataset_name='cifar100', trn_transforms=None, val_transforms=None):
    if trn_transforms is None:
        val_dataset=[]
        if dataset_name=='cifar100':
            train_dataset = datasets.CIFAR100('../data', train=True, download=True,
                        transform = None, )                        
            
        elif dataset_name=='cifar10':
            train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                        transform = None, )                       
            
        elif dataset_name=='ImageNet':
            train_dataset = datasets.ImageNet('../data', 'train', download=True,
                                              transform = None, )
                        
        
    else:
        # data is the name of the folder to be used to sore images
        if dataset_name=='cifar100':
            train_dataset = datasets.CIFAR100('../data', train=True, download=True,
                        transform = trn_transforms, 
                        )                                
            val_dataset = datasets.CIFAR100('../data', train=False, download=True,
                        transform = val_transforms,  )   
        elif dataset_name=='cifar10':
            train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                        transform = trn_transforms, 
                        )                                
            val_dataset = datasets.CIFAR10('../data', train=False, download=True,
                        transform = val_transforms,  )  
        
        elif dataset_name=='ImageNet':
            train_dataset = datasets.ImageNet('../data', split='train', download=True,
                        transform = trn_transforms, 
                        )                                
            val_dataset = datasets.ImageNet('../data', split='val', download=True,
                        transform = val_transforms,  )  
                        
    num_classes = len(train_dataset.classes)
    return  train_dataset,  val_dataset, num_classes                       
        
        

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # Data loading code
    # to use ImageNet data, use the dataloader similar to https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/imagenet/main.py
    dataset_name = args.dataset_name
    train_dataset, _, args.num_classes = get_data(dataset_name=dataset_name)  
    
    image_size = train_dataset[1][0].size
    crop_size = train_dataset[1][0].size
    del train_dataset
    
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    
    img_size = (256, 256)
    crop_size = img_size
    
    trn_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BICUBIC),                
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.5)),                      
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        ])

    val_transforms = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset,  val_dataset, args.num_classes  = get_data(dataset_name=dataset_name, 
                                            trn_transforms=trn_transforms, 
                                            val_transforms=val_transforms)
    
        
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # create model        
    #args.num_classes = len(train_dataset.classes)
    
    if args.arch == 'googlenet':
        model = googlenet(pretrained = False, progress = True, 
                          num_classes = args.num_classes,                           
                          num_epochs = args.epochs,
                          min_kohonen_dim = args.min_kohonen_dim)
    elif args.arch == 'effnet':  #  https://github.com/lukemelas/EfficientNet-PyTorch/
        ''' there are many EffNet models:
                    Name	 # Params	Top-1 Acc.	Pretrained?
            efficientnet-b0	  5.3M	      76.3	       ???
            efficientnet-b1	  7.8M	      78.8	       ???
            efficientnet-b2	  9.2M	      79.8	       ???
            efficientnet-b3	  12M	      81.1	       ???
            efficientnet-b4	  19M	      82.6	       ???
            efficientnet-b5	  30M	      83.3	       ???
            efficientnet-b6	  43M	      84.0	       ???
            efficientnet-b7	  66M	      84.4	       ???
        '''    
    
        if args.pretrained:
                model = EfficientNet.from_pretrained('efficientnet-b0', 
                                 num_classes= args.num_classes, include_top=True)                 
        else:
            model = EfficientNet.from_name('efficientnet-b0', 
                             num_classes= args.num_classes, include_top=True) 
            
    # elif args.arch=='CnnKohonen':      
    #     model = CKCNet(num_classes=args.num_classes, 
    #                    num_epochs=args.epochs, 
    #                    kohonen_dim=args.kohonen_dim,
    #                    max_num_neighbors=args.neighbor_radius, 
    #                    pretrained=args.pretrained)         
    #     '''  I used the default parameters: hidden_units=64,  model_name='efficientnet-b0' '''
                 
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:        
            print("=> creating model '{}'".format(args.arch))            
            model = models.__dict__[args.arch]()                
        model.fc = nn.Linear(512, args.num_classes) # added by rawi, change the output size 
       
            
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    
    print(args)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
   
   
    
    if args.evaluate:
        ''' 
        TODO 
        We need to take care of this later        
        '''
        validate(val_loader, model, criterion, args)
        return
    
    
    for epoch in range(args.start_epoch, args.epochs):   
        
        # start validation at the beggening, to see how the model behaves without training
                
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)         
                
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)              
                              
        # remember best acc@1 and save checkpoint                
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args ):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    
    # switch to train mode
    model.train()
    # pred = []
    # pred = torch.tensor([]).cuda() if torch.cuda.is_available() else torch.tensor([])
           
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        
        # plt.imshow(  images[1,:].permute(1, 2, 0)  ); plt.show() # to see the image
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output 
        if args.arch=='CnnKohonen' or args.arch=='googlenet':
            output = model(images, epoch)     # epoch will control the Kohonent neighbors   
        else:
            output = model(images)           
                        
        # compute loss     
        if args.arch=='googlenet':            
            loss = criterion(output.logits, target)
        else:            
            loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5), args=args)  # this has no effect now
        img_x_size = images.size(0)
        losses.update(loss.item(), img_x_size)        
        top1.update(acc1[0], img_x_size)
        top5.update(acc5[0], img_x_size)        
                

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
                    
    return 

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):                                       
                   
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available(): # no need for cuda as we will not use any computation on targets
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            
            output = model(images, 0) # 0 is used to use all active nodes after training
            
            if args.arch=='googlenet':            
                 loss = criterion(output, target)
            else:            
                loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5), args=args, mode='test')
            
            img_x_size = images.size(0)
            losses.update(loss, img_x_size)
            top1.update(acc1[0], img_x_size)
            top5.update(acc5[0], img_x_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
                
    return top1.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    

def accuracy(output, target, topk=(1,), args=None, mode='train'):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        if args.arch=='googlenet' and mode=='train':
            _, pred= output.logits.topk(maxk, 1, True, True)
        else:
           _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()





''' 
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training

'''