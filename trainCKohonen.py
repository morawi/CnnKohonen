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

import matplotlib.pyplot as plt
# from audtorch.metrics.functional import pearsonr
from torch_cosine import cosine_similarity_n_space

from scipy.ndimage import gaussian_filter1d
import numpy as np

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
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument("--redirect-std-to-file", default=False, type=lambda x: (str(x).lower() == 'true'),  help="True/False - default False; if True sets all console output to file")
parser.add_argument("--experiment-name", type=str, default='results', help="name of the folder inside saved_models")
parser.add_argument('--pretrained', default=False, type=lambda x: (str(x).lower() == 'true'),  help='use pre-trained model')


best_acc1 = 0
error_metric = nn.MSELoss(reduction='mean')


def main():
    args = parser.parse_args()    
    
    if platform.system() == 'Windows': 
        args.workers= 0 # change to 0 if there's a problem
    # used for debugging    
    # args.batch_size = 16
    # args.print_freq = 200
    # args.epochs=30
    # args.arch = 'effnet'
    # args.pretrained=False
    args.max_num_neighbors=10
    args.dataset_name = 'cifar10'
    
    
    args.kohonen_frequency = args.epochs//5 # use very high number to disable it .. if 5 ...means use kohonen every 5 epochs
        
    if  args.redirect_std_to_file:    # sending output to file                
        out_file_name = args.dataset_name + str(datetime.now().ctime())
        print('Output sent to ', out_file_name)
        sys.stdout = open(out_file_name+'.txt',  'w')
    
    print(args)
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
            val_dataset = datasets.CIFAR100('../data', split='val', download=True,
                        transform = val_transforms,  )  
                        
            
    return  train_dataset,  val_dataset                          
        
        

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
    train_dataset, _ = get_data(dataset_name=dataset_name)  
    
    image_size = train_dataset[1][0].size
    crop_size = train_dataset[1][0].size
    del train_dataset
    
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    
    trn_transforms = transforms.Compose([
        # transforms.Resize(image_size, interpolation=Image.BICUBIC),
           transforms.RandomResizedCrop(image_size, scale=(0.5, 1.5)),                      
           transforms.RandomHorizontalFlip(),
           transforms.ToTensor(),
           normalize,
           ])

    val_transforms = transforms.Compose([
        # transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    train_dataset,  val_dataset  = get_data(dataset_name=dataset_name, 
                                            trn_transforms=trn_transforms, 
                                            val_transforms=val_transforms)
    
    args.num_outputs = len(train_dataset.classes)    
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
    args.num_classes = len(train_dataset.classes)
    
    
    if args.arch == 'effnet':  #  https://github.com/lukemelas/EfficientNet-PyTorch/
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
                                 num_classes= args.num_outputs, include_top=True)                 
        else:
            model = EfficientNet.from_name('efficientnet-b0', 
                             num_classes= args.num_outputs, include_top=True) 
        
    else:
        if args.pretrained:
            print("=> using pre-trained model '{}'".format(args.arch))
            model = models.__dict__[args.arch](pretrained=True)
        else:        
            print("=> creating model '{}'".format(args.arch))            
            model = models.__dict__[args.arch]()                
        model.fc = nn.Linear(512, args.num_outputs) # added by rawi, change the output size 
       
            
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
   
        
    ''' Not in use for now
    Lambda    
    1 should be the start input to train,
    0.2 should be the last lambad input to thrain 
    
    Not sure what and how will the Lambda effect be after adding the neighborhood map, the
    latter will contorl the (window size) of winner nodes 
    
    '''
    # max_lambda = 1; min_lambda=0.2 # this will be used as sigma for the Gaussian filter that controls the radius of the winner nodes
    # lambda_values = np.arange(max_lambda, min_lambda, -(max_lambda-min_lambda)/args.epochs)
    
    
    if args.evaluate:
        ''' 
        TODO 
        We need to take care of this later        
        '''
        validate(val_loader, model, criterion, args)
        return
    
    # prev_pred = 99*torch.rand(len(train_dataset), args.num_outputs).cuda() # some random value
    neighbors_map = [1+args.max_num_neighbors*(args.epochs-1-epoch)//args.epochs for epoch in range(0, args.epochs)]
    
    for epoch in range(args.start_epoch, args.epochs):   
        
        # start validation at the beggening, to see how the model behaves without training
        if epoch%2==0:
            validate(val_loader, model, criterion, args)
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, 
                     optimizer, epoch,                      
                     neighbors_map[epoch], 
                     args, 
                     lambda_val=1 #lambda_values[epoch]
                     ) 
        
        # err = error_metric(pred, prev_pred)
        # print('Error between current and prev output ', err)        
        # prev_pred = pred # storing the pred to be compared next time       

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


def train(train_loader, model, criterion, optimizer, epoch, 
          neighbor_idx, args, lambda_val=1 ):
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
        output = model(images)        
        # compute loss
        loss = criterion(output, target)   
        # if epoch%args.kohonen_frequency==0: # the usual way should be 0, put 000099 to skip         
        #     loss = criterion(output, target)   
        # else: # get winner nodes according to neighborhood defined via neighbor_idx
            
        #     winner_nodes = get_winners(output.shape, torch.argmax(output, dim=1), 
        #                                lambda_val, neighbor_idx=neighbor_idx)                    
        #     loss = winner_nodes*output        
        #     loss = abs(loss).mean()            
        #     # pred = torch.cat((pred, output.detach()))

        # measure accuracy and record loss
        acc1, acc5, _ = accuracy(output, target, topk=(1, 5))  # this has no effect now
        # acc1=[0]; acc5=[0] # using dummy values, so that not to disrubt the whole code
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
    pred = torch.tensor([]).cuda() if torch.cuda.is_available() else torch.tensor([])
    all_targets = torch.tensor([])
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):                                       
                   
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available(): # no need for cuda as we will not use any computation on targets
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            

            # measure accuracy and record loss
            acc1, acc5, best_5_output = accuracy(output, target, topk=(1, 5))
            pred = torch.cat((pred, output.detach())) # use output.detach() for all outputs instead of best_5_output
            
            img_x_size = images.size(0)
            losses.update(loss, img_x_size)
            top1.update(acc1[0], img_x_size)
            top5.update(acc5[0], img_x_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            all_targets=torch.cat((all_targets, target.cpu().float()))

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
        
        pred=pred.cpu() # it has to be on the cpu so that it is GPU batched in the cosine
        dist_mat, idx_per_sample = cosine_similarity_n_space(pred,  
                                          dist_batch_size=1000)
        find_group_accuracy_via_cosine_dist(all_targets, idx_per_sample, args)
        
                
    return top1.avg

def find_group_accuracy_via_cosine_dist(all_targets, idx_per_sample, args):
    target_map = all_targets[idx_per_sample]
    new_grouping={}
    original_grouping={}
    for i in range(0, args.num_classes):
        new_grouping[i] = torch.nonzero(target_map==i)
        original_grouping[i] = torch.nonzero(all_targets==i)
        
    target_match = []; 
    idx_to_search = list(range(0, args.num_classes))
    for i in range(0, args.num_classes):
        zz=[]
        for jj in idx_to_search:                
            xx = np.intersect1d(new_grouping[i].numpy(), 
                                original_grouping[jj].numpy())
            zz.append(len(xx))
        idx_score = np.argmax(np.array(zz))
        idx_to_search.pop(idx_score)
        target_match.append(zz[idx_score])
    print('Cosiene Dist Group Label Acc', 100*sum(target_match)/len(all_targets) )

def get_winners(output_shape, winners_idx=None, lambda_val=1, neighbor_idx=2):
    
    num_samples = len(winners_idx) # num_samples = output_shape[0]
    x_idx = torch.arange(0, num_samples)
    winner_nodes = torch.zeros(output_shape).cuda()
    if neighbor_idx>0: 
        for n_idx in range(1, neighbor_idx, 1):
            winners_next = winners_idx + n_idx*(winners_idx<(output_shape[1]-n_idx)) # (winners_idx+2)%output_shape[1] # % used to keep the values within the upper bound
            winners_prev = winners_idx - n_idx 
            winners_prev *= winners_prev>0 # in case we have a -ve value, set them to zero     # a *= (a>0)   # https://stackoverflow.com/questions/3391843/how-to-transform-negative-elements-to-zero-without-a-loop
            # now, setting amplitiudes of the neigbors and the max node
            winner_nodes[x_idx, winners_next] = 0.8 
            winner_nodes[x_idx, winners_prev] = 0.8 
    winner_nodes[x_idx, winners_idx] = 1 # 100% power to the winner node
    
    # not sure if we should remove this gaussian filter. not yet.
    winner_nodes=gaussian_filter1d(winner_nodes.cpu().numpy(), lambda_val) 
    winner_nodes= torch.tensor(winner_nodes).cuda() 
    
    # we need to normalize the amplitude as the Gaussian has reduced it
    v_max, _ = torch.max(winner_nodes, dim=1)        
    winner_nodes = torch.transpose(winner_nodes, 0, 1)/ v_max # torch.div(winner_nodes, v)
    winner_nodes = torch.transpose(winner_nodes, 0, 1)
    
    return winner_nodes
    


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
    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))    
    """Sets the learning rate to the initial LR decayed by 0.9 every 31 epochs"""
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def my_prediction(output):
    
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)    
    
    return pred.float()

    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        
        best_5_output = output.gather(1, pred)
        
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        res.append(best_5_output)
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