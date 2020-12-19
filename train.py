from data_preprocessing.train_val_split import split_train_val_5_fold
from dataset.dataset import dataloader
from dataset.augmentations import tensor_batch2PIL
import numpy as np
import timm
from PIL import Image
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import math
import shutil
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
        self.ce = torch.nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.reduction:
            return loss.mean()
        else:
            return loss


def create_model(pretrained=False):
    return timm.create_model('inception_resnet_v2', pretrained=pretrained, num_classes=5)


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


def init_device_pytorch():
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        gpus = list(range(torch.cuda.device_count()))
        print('=> using GPU devices: {}'.format(', '.join(map(str, gpus))))
    else:
        gpus = None
        print('=> using CPU device')
    device = torch.device('cuda:{}'.format(gpus[0])) if gpus else torch.device('cpu')
    cudnn.benchmark = True
    return gpus, device


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(exp_dir, base_name, state, is_best=False):
    """ Saves a model's checkpoint.
    :param exp_dir: Experiment directory to save the checkpoint into.
    :param base_name: The output file name will be <base_name>_latest.pth and optionally <base_name>_best.pth
    :param state: The model state to save.
    :param is_best: If True <base_name>_best.pth will be saved as well.
    """
    filename = os.path.join(exp_dir, base_name + '_latest.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(exp_dir, base_name + '_best.pth'))


if __name__ == '__main__':

    ROOT = '/home/noteme/PycharmProjects/comp/data/cassava-disease'
    OUT_DIR = os.path.join(ROOT, 'experiment')

    # create output dir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    SEED = 1337
    BATCH_SIZE = 32
    lr = 1e-4
    epoches = 6
    folds, images_list = split_train_val_5_fold(root=ROOT, seed=SEED)
    gpus, device = init_device_pytorch()

    for key in folds.keys():
        # initiate datasets
        dataset_train = dataloader(split=folds[key]['train'], images_list=images_list, batch_size=BATCH_SIZE)
        dataset_val = dataloader(split=folds[key]['val'], images_list=images_list, batch_size=BATCH_SIZE, train=False)

        # initiate model
        model = create_model(pretrained=True)
        model = model.to(device)

        # initiate loss
        loss_fn = FocalLoss()

        # initiate optimizer in order to control dynamics of learning rate for each layer
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)  # switch to SGD

        # initiate scheduler in order to control global lr
        scheduler = CosineAnnealingLR(optimizer, T_max=math.floor(len(dataset_train)/BATCH_SIZE), eta_min=1e-6)  # switch to Every N epoch Scheduler

        # initiate tensorboard logger
        logger = SummaryWriter(OUT_DIR)

        # init best metric for proper checkpoint save
        best_metric = 0

        # init total iterations
        total_iter = 0

        # add early stopper
        val_count = 0
        current_val_loss = 0

        for epoch in range(epoches):

            # initiate running metrics
            run_train_loss = AverageMeter('train_loss')
            run_train_acc = AverageMeter('train_acc')
            run_val_loss = AverageMeter('val_loss')
            run_val_acc = AverageMeter('val_acc')

            # train
            model.train()
            pbar = tqdm(dataset_train, total=len(dataset_train))
            for images, labels in pbar:
                # get model predictions
                images = images.to(device)
                labels = labels.to(device)

                # update optimizer params
                optimizer.zero_grad()

                predictions = model(images)

                # calculate loss
                train_loss = loss_fn(predictions, labels)

                # backpropogation
                train_loss.backward()

                optimizer.step()

                # calculate acc
                acc1 = accuracy(predictions, labels)

                # update metrics
                run_train_loss.update(train_loss.item(), images.size(0))
                run_train_acc.update(acc1[0].item(), images.size(0))

                # update logger
                total_iter += BATCH_SIZE

                # Batch logs
                pbar.set_description(
                    'TRAINING: Current epoch: {}; '
                    'Loss {loss.val:.4f} ({loss.avg:.4f}); '
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, loss=run_train_loss, top1=run_train_acc))

                logger.add_scalars('train', {'loss': run_train_loss.val}, total_iter)
                logger.add_scalars('train/acc', {'top1': run_train_acc.val}, total_iter)

            # Epoch logs
            logger.add_scalars('epoch/acc', {'train': run_train_acc.avg}, epoch)

            # validation
            model.eval()
            with torch.no_grad():
                pbar = tqdm(dataset_val, total=len(dataset_val))
                for images, labels in pbar:
                    images = images.to(device)
                    labels = labels.to(device)

                    predictions = model(images)

                    # calculate loss
                    val_loss = loss_fn(predictions, labels)

                    # calculate acc
                    acc_1 = accuracy(predictions, labels)

                    # update metrics
                    run_val_loss.update(val_loss.item(), images.size(0))
                    run_val_acc.update(acc_1[0].item(), images.size(0))

                    # Batch logs
                    pbar.set_description(
                        'VALIDATION: Current epoch: {}; '
                        'Loss {loss.val:.4f} ({loss.avg:.4f}); '
                        'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, loss=run_val_loss, top1=run_val_acc))

                # Epoch logs
                logger.add_scalars('epoch/acc', {'val': run_val_acc.avg}, epoch)

            # save best model and latest model with all required parameteres
            if run_val_acc.avg > best_metric:
                best_metric = run_val_acc.avg
                save_best = True
            else:
                save_best = False

            save_checkpoint(OUT_DIR, 'model', {
                'epoch': epoch,
                'state_dict': model.module.state_dict() if gpus and len(gpus) > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, save_best)
