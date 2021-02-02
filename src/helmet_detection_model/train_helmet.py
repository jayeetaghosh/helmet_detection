import logging
logging.basicConfig(filename='/home/ec2-user/SageMaker/helmet_detection/model/model_helmet.log',level=logging.INFO)
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from datetime import datetime
import time
from glob import glob
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import RandomSampler, SequentialSampler
import utils as utils_torchvision
import transforms as T
from engine import train_one_epoch, evaluate
from PIL import Image
from PIL import ImageDraw
import warnings

warnings.filterwarnings("ignore")
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)




class DatasetHelmet(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        
        image, boxes, labels = self.load_image_and_boxes(index)
        num_boxes = len(boxes)
        if num_boxes > 0:
            target = {}
            new_boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((num_boxes,), dtype=torch.int64)
            area = (new_boxes[:, 3] - new_boxes[:, 1]) * (new_boxes[:, 2] - new_boxes[:, 0])
            # suppose all instances are not crowd 
            iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

            target['boxes'] = new_boxes
            target['labels'] = labels
            target['image_id'] = torch.tensor([index])
            target["area"] = area
            target["iscrowd"] = iscrowd
        else:
            print("no boxes")
            target = {}

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        TRAIN_ROOT_PATH = args.train + "images"
        image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}', cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['image'] == image_id]
        boxes = records[['left', 'top', 'width', 'height']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        labels = records['label'].values
        return image, boxes, labels

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.ImgAug())
    return T.Compose(transforms)

def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
    # Define which device to use, either gpu or cpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        model.to(device)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model
    


def main(args):
#     Read images label csv file
    image_labels = pd.read_csv('/home/ec2-user/SageMaker/helmet_detection/input/image_labels.csv')#.fillna(0)

    # #     Split annotations into train and validation
    np.random.seed(0)
    image_names = np.random.permutation(image_labels.image.unique())
    valid_image_len = int(len(image_names)*0.2)
    images_valid = image_names[:valid_image_len]
    images_train = image_names[valid_image_len:]    
# #     Get a small dataset to try 
#     images_valid = images_valid[:20]
#     images_train = images_train[:80]
    
    logging.info(f"images_valid {images_valid}, \n images_train {images_train}")
    # Define train and validation datasets and data loaders
    TRAIN_ROOT_PATH = args.train 

    train_dataset = DatasetHelmet(
        image_ids=images_train,
        marking=image_labels,
        transforms=get_transform(train=True),
        test=False,
    )
    validation_dataset = DatasetHelmet(
        image_ids=images_valid,
        marking=image_labels,
        transforms=get_transform(train=False),
        test=True,
    )    

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
        collate_fn=utils_torchvision.collate_fn
    )
    data_loader_valid = torch.utils.data.DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
        collate_fn=utils_torchvision.collate_fn
    )
    print(f"We have {len(train_dataset)} images for training and {len(validation_dataset)} for validation")
    
    # Set up model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ## Our dataset has two classes only - helmet and not helmet
    num_classes = 2
    ## Get the model using our helper function
    model = get_model(num_classes)
    print(f"Loaded model")

    # Set up training
    start_epoch = 0
    end_epoch = args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    print(f"Loaded model parameters")

    ## if retraining from a checkpoint file
    if args.retrain:
        
        checkpoint = torch.load(os.path.join(args.model_dir, "model_checkpoint.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        end_epoch = start_epoch + args.epochs
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
       
    print(start_epoch, end_epoch)

    # Train model
    loss_epoch = []
    
    for epoch in range(start_epoch, end_epoch):
        # train for one epoch, printing every 1 iterations
        print(f"Training epoch {epoch}")
        train_one_epoch(model, optimizer, data_loader, data_loader_valid, device, epoch, loss_epoch, print_freq=1)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, data_loader_valid, device=device, print_freq=1)
        # save checkpoint model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.model_dir, "model_checkpoint.pt"))
        

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model_helmet_frcnn.pt"))
    loss_df = pd.DataFrame(loss_epoch, columns=["train_loss", "val_loss"])
    loss_df.reset_index(inplace=True)
    loss_df = loss_df.rename(columns = {'index':'Epoch'})
    print(loss_df)
    loss_df.to_csv (os.path.join(args.model_dir, "loss_epoch.csv"), index = False, header=True)

                                                

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description="Train or Retrain helmet detection model.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train, default 10")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size, default 4")
    parser.add_argument('--retrain', type=bool, default=False, help="Whether to retrain from a checkpoint file, default False")


    # Data, model, and output directories
    if 'SM_MODEL_DIR' in os.environ:
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'],
                           help="Model output directory")
    else:
        parser.add_argument('--model-dir', type=str, default='/home/ec2-user/SageMaker/helmet_detection/model/',
                           help="Model output directory")
    if 'SM_CHANNEL_TRAIN' in os.environ:
        parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'],
                           help="Train directory with image files saved within 'images' and annotations saved within 'train.json' ")
    else:
        parser.add_argument('--train', type=str, default='/home/ec2-user/SageMaker/helmet_detection/input/',
                           help="Train directory with image files saved within 'images' and annotations saved within 'train.json' ")

    args, _ = parser.parse_known_args()

    main(args)
