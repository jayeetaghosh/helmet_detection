import random
import torch

from torchvision.transforms import functional as F

import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

    
class ImgAug(object):
    def __call__(self, image, target):
        # Define augmentations
        seq = iaa.Sequential([
            iaa.GammaContrast((0.9, 1.1)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                       scale=(0.5, 1.5),
                       rotate=(-5, 5),
                       shear=(-5, 5))
        ])

        # Convert images to numpy uint8 HWC BGR
        img = image.numpy()
        img = img*255
        img = img.astype(np.uint8)
        img = np.moveaxis(img, 0, 2) # CHW to HWC
        img = img[:,:,::-1] # RGB to BGR

        # Convert bounding boxes to numpy then conver to ImgAug BoundingBoxes
        boxes = target['boxes']#.numpy() # no need?
        bbs = BoundingBoxesOnImage([BoundingBox(*b) for b in boxes],
                                   shape=img.shape)

        # Apply augmentations
        img, bbs = seq(image=img, bounding_boxes=bbs)

        # Convert images back to Tensor float32 CHW RGB
        img = img[:,:,::-1] # BGR to RGB
        img = np.moveaxis(img, 2, 0) # HWC to CHW
        img = img.astype(np.float32)
        img = img/255
        image = torch.from_numpy(img)

        # Convert boxes back to Tensor
        boxes = np.array([[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs])
        target['boxes'] = torch.from_numpy(boxes)

        return image, target
