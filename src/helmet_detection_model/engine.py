import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, data_loader, data_loader_test, device, epoch, loss_epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    ##########################    
    # Run iterations on training dataset
    ##########################    

    running_loss_train = 0.0

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) # Get dictionary of tensor of losses - 
#         print("****loss_dict", loss_dict)
        
        losses = sum(loss for loss in loss_dict.values())
#         print("****losses", losses) # Get total loss
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict) # Only one gpu used, so same as loss_dict
#         print("****loss_dict_reduced", loss_dict_reduced)
        
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
#         print("****losses_reduced", losses_reduced) # total loss, only one gpu, so same as losses
        
        loss_value = losses_reduced.item()
#         print("****loss_value", loss_value)
        
        # add total loss into running_loss
        # https://discuss.pytorch.org/t/plotting-loss-curve/42632/4
        running_loss_train += loss_value * len(images)
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()# Ideally this should be losses_reduced.backward(), incase multigpu was used
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
            
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])            
 
    # Get train loss per epoch
    train_loss_epochs = running_loss_train/len(data_loader.dataset)
    print("****train_loss_epochs", train_loss_epochs, len(data_loader.dataset))
    metric_logger.add_meter("train_loss_epochs", train_loss_epochs)
    
    ##########################    
    # Get validation loss by model in train mode and gradient deactivated appended to metric_logger
    ##########################    

    # https://discuss.pytorch.org/t/compute-validation-loss-for-faster-rcnn/62333
    header = 'Val Epoch: [{}]'.format(epoch)
    running_loss_val = 0
    for images, targets in metric_logger.log_every(data_loader_test, print_freq, header):
        print("within test")
#         images = [image.to(device) for image in images]
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            val_loss_dict = model(images, targets)
            val_losses = sum(loss for loss in val_loss_dict.values())
            
            # reduce losses over all GPUs for logging purposes
            val_loss_dict_reduced = utils.reduce_dict(val_loss_dict)
            val_losses_reduced = sum(loss for loss in val_loss_dict_reduced.values())
            
            val_loss_value = val_losses_reduced.item()
#             print("****val_loss_value", val_loss_value)
            # add total loss into running_loss
            running_loss_val += val_loss_value * len(images)
            
            metric_logger.update(loss=val_losses_reduced, **val_loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # Get validation loss per epoch 
    val_loss_epochs = running_loss_val/len(data_loader_test.dataset)
    print("****val_loss_epochs", val_loss_epochs, len(data_loader_test.dataset))
    metric_logger.add_meter("val_loss_epochs", val_loss_epochs)
    
    loss_epoch.append([train_loss_epochs, val_loss_epochs])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, print_freq):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, print_freq, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
