import argparse
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import cv2
import json

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# from utils import utils


class FramesDataset(Dataset):
    """Creates a dataset that can be fed into DatasetLoader

    Args:
        frames (list): A list of cv2-compatible numpy arrays or
          a list of PIL Images
    """
    def __init__(self, frames):
        # Convert to list of tensors  
        
        x = [F.to_tensor(img) for img in frames] 
        # Define which device to use, either gpu or cpu
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Send the frames to device
        x_device = [img.to(device) for img in x]

        self.x = x_device #x

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)


class ObjectDetector():
    """ObjectDetector class with staticmethods that can be called from outside by importing as below:
    from helmet_detector.detector import ObjectDetector
    
    The staic methods can be accessed using ObjectDetector.<name of static method>()

    """
    

    @staticmethod
    def load_custom_model(model_path=None, num_classes=None):
        """Load a model from local file system with custom parameters

        Load FasterRCNN model using custom parameters

        Args:
            model_path (str): Path to model parameters
            num_classes (int): Number of classes in the custom model
        Returns:
            model: Loaded model in evaluation mode for inference
        """
        # load an object detection model pre-trained on COCO
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes)
        
        # load previously fine-tuned parameters
        # Define which device to use, either gpu or cpu
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path))
            model.to(device)
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        # Put the model in evaluation mode
        model.eval()

        return model
    

    @staticmethod
    def run_detection(img, loaded_model):
        """ Run inference on single image

        Args:
            img: image in 'numpy.ndarray' format
            loaded_model: trained model
        Returns:
            Default predictions from trained model
        """

        # need to make sure we have 3d tensors of shape [C, H, W]
        with torch.no_grad():
            prediction = loaded_model(img)

        return prediction


    @staticmethod
    def to_dataframe_highconf(predictions, conf_thres, frame_id):
        """ Converts the default predictions into a Pandas DataFrame, only predictions with score greater than conf_thres

        Args:
            predictions (list): Default FasterRCNN implementation output.
              This is a list of dicts with keys ['boxes','labels','scores']
            frame_id : frame id
            conf_thres: score greater than this will be kept as detections
        Returns:
            A Pandas DataFrame with columns
              ['frame_id','class_id','score','x1','y1','x2','y2']
        """
        df_list = []
        for i, p in enumerate(predictions):
            boxes = p['boxes'].detach().cpu().tolist()
            labels = p['labels'].detach().cpu().tolist()
            scores = p['scores'].detach().cpu().tolist()
            df = pd.DataFrame(boxes, columns=['x1','y1','x2','y2'])
            df['class_id'] = labels
            df['score'] = scores
            df['frame_id'] = frame_id
            df_list.append(df)
        df_detect = pd.concat(df_list, axis=0)
        df_detect = df_detect[['frame_id','class_id','score','x1','y1','x2','y2']]
        
        # Keep predictions with high confidence, with score greater than conf_thres
        df_detect = df_detect.loc[df_detect['score'] >= conf_thres]
        return df_detect


    @staticmethod
    def to_dataframe(predictions):
        """ Converts the default predictions into a Pandas DataFrame

        Args:
            predictions (list): Default FasterRCNN implementation output.
              This is a list of dicts with keys ['boxes','labels','scores']
        Returns:
            A Pandas DataFrame with columns
              ['frame_id','class_id','score','x1','y1','x2','y2']
        """
        df_list = []
        for i, p in enumerate(predictions):
            boxes = p['boxes'].detach().cpu().tolist()
            labels = p['labels'].detach().cpu().tolist()
            scores = p['scores'].detach().cpu().tolist()
            df = pd.DataFrame(boxes, columns=['x1','y1','x2','y2'])
            df['class_id'] = labels
            df['score'] = scores
            df['frame_id'] = i
            df_list.append(df)
        df_detect = pd.concat(df_list, axis=0)
        df_detect = df_detect[['frame_id','class_id','score','x1','y1','x2','y2']]
        return df_detect

    @staticmethod
    def calc_iou(boxA, boxB):
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou


    @staticmethod
    def evaluate_detections_iou(gt, det, iou_threshold):
        """Evaluate and obtain FN and FP records between detection and annotations

        Args:
            df_detect (pandas.DataFrame): Detected boxes in a Pandas Dataframe
              with columns ['frame_id','class_id','score','x1','y1','x2','y2']
            df_annot (pandas.DataFrame): Known/annotation boxes in a Pandas
              Dataframe with columns ['frame_id','class_id','x1','y1','x2','y2']

        Returns:
            result (pandas.DataFrame): Count of total number of objects in gt and det, and tp, fn, fp
              with columns ['num_object_gt', 'num_object_det', 'tp', 'fn', 'fp']
            df_fn (pandas.DataFrame): False negative records in a Pandas Dataframe
              with columns ['frame_id','class_id','x1','y1','x2','y2']
            df_fp (pandas.DataFrame): False positive records in a Pandas Dataframe
              with columns ['frame_id','class_id', 'score', 'x1','y1','x2','y2']
        """
        
        if (gt is not None) and (det is not None):
            matched = []
             
            for g in range(gt.shape[0]):
                count = 0
                for d in range(det.shape[0]):
                    iou = ObjectDetector.calc_iou(np.array(gt.iloc[g,2:]), np.array(det.iloc[d,3:]))
                    
                    if (iou > iou_threshold):
                        if (count == 0):
                            max_conf = det.iloc[d,2]
                            temp = [g,d,iou, det.iloc[d,2]]                        
                            count +=1
                        elif (count > 0):
                            print("Multiple detections found, keep only with highest confidence") 
                            if (max_conf < det.iloc[d,2]):
                                max_conf = det.iloc[d,2]
                                temp = [g,d,iou, det.iloc[d,2]]
                                count +=1
                            
                if (count != 0):
                    matched.append(temp)
                        
            df_tp = pd.DataFrame(matched, columns = ['gt_index', 'det_index', 'iou', 'det_conf'])
            
            # To qualitatively find detection error, output fn and fp boxes. just visualize them on the frame
            # Get unmatched gt - these are FNs
            df_fn = []
            num_fn = 0
            for i in range(gt.shape[0]):
                
                if i not in df_tp['gt_index'].tolist():
                    df_fn.append(gt.iloc[i,:])
                    num_fn +=1
            if num_fn > 0:
                df_fn = pd.DataFrame(data=df_fn)
                df_fn.columns = ['frame_id','class_id','x1','y1','x2','y2']
            else:
                df_fn = None
            
            # Get unmatched det - these are FPs
            df_fp = []
            num_fp = 0
            for i in range(det.shape[0]):
                if i not in df_tp['det_index'].tolist():
                    df_fp.append(det.iloc[i,:])
                    num_fp +=1

            if num_fp > 0:
                df_fp = pd.DataFrame(data=df_fp)
                df_fp.columns = ['frame_id','class_id', 'score', 'x1','y1','x2','y2']
            else:
#                 print("num_fp = 0 in frame_id {}".format(gt.iloc[0,0]))
                df_fp = None

            # To quantify detection error, output number of helmets in gt, number of helmets in det, tp, fn, fp
            frame_id = gt.iloc[0,0]
            tp = len(df_tp['gt_index'].unique())
            result = []
            result.append([frame_id,
                      gt.shape[0], 
                      det.shape[0],
                      tp,
                      num_fn, 
                      num_fp])
            result = pd.DataFrame(data=result, columns = ['frame_id', 'num_object_gt', 'num_object_det', 'tp', 'fn', 'fp'])
    
            
        else:
            result = None
            df_fn = None
            df_fp = None


        return result, df_fn, df_fp


    @staticmethod
    def find_frames_high_fn_fp(eval_det, fn_thres, fp_thres):
        """ Find frames with high fn and fp, fn >= fn_thres and fp >= fp_thres
        Arg:
            eval_det: Detection evaluation matrix for whole play
            fn_thres: Get a list of frames where fn is greater than equal to this value
            fp_thres: Get a list of frames where fn is greater than equal to this value
        Return: 
            frame_list: List of frames with high fn and fp
        
        """
        frame_list = eval_det[(eval_det['fn'] >= fn_thres) & (eval_det['fp'] >= fp_thres)]['frame_id'].tolist()
        return frame_list
    

    @staticmethod
    def run_detection_video(video_in, model_path, full_video=True, subset_video=60, conf_thres=0.9):
        """ Run detection on video

        Args:
            video_in: Input video path
            model_path: Location of the pretrained model.pt 
            full_video: Bool to indicate whether to run the whole video, default = False
            subset_video: Number of frames to run detection on
            conf_thres = Only consider detections with score higher than conf_thres, default = 0.9
        Returns:
            Predicted detection for all the frames in a video
            df_predictions (pandas.DataFrame): prediction of detected object for all frames 
              with columns ['frame_id', 'class_id', 'score', 'x1', 'y1', 'x2', 'y2']
            
        """
        # Capture the input video
        vid = cv2.VideoCapture(video_in)

        # Get video title
        vid_title = os.path.splitext(os.path.basename(video_in))[0]

        # Get total number of frames
        num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # load model 
        num_classes = 2
        model = ObjectDetector.load_custom_model(model_path=model_path, num_classes=num_classes)
        
        # if running for the whole video, then change the size of subset_video with total number of frames 
        if full_video:
            subset_video = int(num_frames)   
                    
        df_predictions = [] # predictions for whole video
        
        for i in range(subset_video): #383

            ret, frame = vid.read()
            print("Processing frame#: {} running detection for videos".format(i))
            
            # Get detection for this frame
            list_frame = [frame]
            dataset_frame = FramesDataset(list_frame)
            prediction = ObjectDetector.run_detection(dataset_frame, model)
            df_prediction = ObjectDetector.to_dataframe_highconf(prediction, conf_thres, i)
            df_predictions.append(df_prediction)
        
        # Concatenate predictions for all frames of the video
        df_predictions = pd.concat(df_predictions)

        return df_predictions
    

    @staticmethod
    def run_detection_frames(frames, model_path, batch_size=4, conf_thres=0.9, start_frame=0, end_frame=-1):
        """ Run detection on list of frames

        Args:
            frames: List of frames between start_frame and end_frame of a full play video
            model_path: Location of the pretrained model.pt 
            batch_size (int): Size of inference minibatch --> not sure we need this
            conf_thres: Only consider detections with score higher than conf_thres, default = 0.9
            start_frame: First frame number to output. Default is 0.
            end_frame: Last frame number to output. If less than 1 then take all frames
        Returns:
            Predicted detection for all the frames between start_frame and end_frame of a full play video
            df_predictions (pandas.DataFrame): prediction of detected object for all frames 
              with columns ['frame_id', 'class_id', 'score', 'x1', 'y1', 'x2', 'y2']
        
        Todo:
            Figure out how reduce confusion around start_frame/end_frame var collision with utils.frames_from_video()
        """
        if end_frame>=1:
            assert start_frame<=end_frame
        if end_frame < 0:
            end_frame = start_frame + len(frames) -1
        # load model 
        num_classes = 2
        model = ObjectDetector.load_custom_model(model_path=model_path, num_classes=num_classes)

        df_predictions = [] # predictions for all frames
        count = 0
        for i in range(start_frame, end_frame):        
            # Get detection for this frame
            dataset_frame = FramesDataset([frames[count]])
            prediction = ObjectDetector.run_detection(dataset_frame, model)
            df_prediction = ObjectDetector.to_dataframe_highconf(prediction, conf_thres, i)
            df_predictions.append(df_prediction)
            count+=1
            
#         dataset = FramesDataset(frames)
#         batcher = DataLoader(dataset, batch_size=batch_size, shuffle=False)
#             for batch in batcher:
#                 prediction = ObjectDetector.run_detection(batch, model)
#                 df_prediction = ObjectDetector.to_dataframe_highconf(prediction, conf_thres, batch)
#                 df_predictions.append(df_prediction)
        
        # Concatenate predictions for all frames of the video
        df_predictions = pd.concat(df_predictions)

        return df_predictions
    
    @staticmethod
    def get_gt_frame(frame_id, cur_boxes):
        """Get ground truth annotations on the frames

        Args:
            frame_id: Frame id 
            cur_boxes: Current annotation boxes "left", "width", "top", "height"
        Returns:
            box_ret: ground truth boxes in a Pandas
              Dataframe with columns ['frame_id','class_id','x1','y1','x2','y2']

        """

        box_out = []
        for box in cur_boxes:
            box_out.append([frame_id, 1, box[0],box[2],box[0]+box[1], box[2]+box[3]])

        # Return gt dataframe
        box_ret = pd.DataFrame(data = box_out, columns = ['frame_id','class_id','x1','y1','x2','y2'])
        return box_ret
    
    
    @staticmethod
    def run_detection_eval_video(video_in, gtfile_name, model_path, full_video=True, subset_video=60, conf_thres=0.9, iou_threshold = 0.5):
        """ Run detection on video

        Args:
            video_in: Input video path
            gtfile_name: Ground Truth annotation json file name
            model_path: Location of the pretrained model.pt 
            full_video: Bool to indicate whether to run the whole video, default = False
            subset_video: Number of frames to run detection on
            conf_thres = Only consider detections with score higher than conf_thres, default = 0.9
            iou_threshold = Match detection with ground trurh if iou is higher than iou_threshold, default = 0.5
        Returns:
            Predicted detection for all the frames in a video, evaluation for detection, a dataframe with bounding boxes for 
            false negatives and false positives
            df_predictions (pandas.DataFrame): prediction of detected object for all frames 
              with columns ['frame_id', 'class_id', 'score', 'x1', 'y1', 'x2', 'y2']
            eval_results (pandas.DataFrame): Count of total number of objects in gt and det, and tp, fn, fp for all frames
              with columns ['frame_id', 'num_object_gt', 'num_object_det', 'tp', 'fn', 'fp']
            fns (pandas.DataFrame): False negative records in a Pandas Dataframe for all frames
              with columns ['frame_id','class_id','x1','y1','x2','y2'], return empty dataframe if no false negatives 
            fps (pandas.DataFrame): False positive records in a Pandas Dataframe for all frames
              with columns ['frame_id','class_id', 'score', 'x1','y1','x2','y2'], return empty dataframe if no false positives 
            
        """
        # Capture the input video
        vid = cv2.VideoCapture(video_in)

        # Get video title
        vid_title = os.path.splitext(os.path.basename(video_in))[0]

        # Get total number of frames
        num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        print("********** Num of frames", num_frames)
        
        # load model 
        num_classes = 2
        model = ObjectDetector.load_custom_model(model_path=model_path, num_classes=num_classes)
        print("Pretrained model loaded")

        # Get GT annotations
        gt_labels = pd.read_csv('/home/ec2-user/SageMaker/0Artifact/helmet_detection/input/train_labels.csv')#.fillna(0)
        video = os.path.basename(video_in)
        print("Processing video: ",video)
        labels = gt_labels[gt_labels['video']==video]

        
        # if running for the whole video, then change the size of subset_video with total number of frames 
        if full_video:
            subset_video = int(num_frames)   
            
        
#         frames = []
        df_predictions = [] # predictions for whole video
        eval_results = [] # detection evaluations for the whole video 
        fns = [] # false negative detections for the whole video 
        fps = [] # false positive detections for the whole video 
        
        for i in range(subset_video): 

            ret, frame = vid.read()
            print("Processing frame#: {} running detection and evaluation for videos".format(i+1))
            
            # Get detection for this frame
            list_frame = [frame]
            dataset_frame = FramesDataset(list_frame)
            prediction = ObjectDetector.run_detection(dataset_frame, model)
            df_prediction = ObjectDetector.to_dataframe_highconf(prediction, conf_thres, i)
            df_predictions.append(df_prediction)

            # Get label for this frame
            cur_label = labels[labels['frame']==i+1] # get this frame's record
            cur_boxes = cur_label[['left','width','top','height']].values
            gt = ObjectDetector.get_gt_frame(i+1, cur_boxes)
            
            
            # Evaluate detection for this frame
            eval_result, fn, fp = ObjectDetector.evaluate_detections_iou(gt, df_prediction, iou_threshold)
            eval_results.append(eval_result)
            if fn is not None:
                fns.append(fn)
            if fp is not None:
                fps.append(fp)
        
        # Concatenate predictions, evaluation resutls, fns and fps for all frames of the video
        df_predictions = pd.concat(df_predictions)
        eval_results = pd.concat(eval_results)
        # Concatenate fns if not empty, otherwise create an empty dataframe
        if not fns:
            fns = pd.DataFrame()
        else:
            fns = pd.concat(fns)
        # Concatenate fps if not empty, otherwise create an empty dataframe
        if not fps:
            fps = pd.DataFrame()
        else:
            fps = pd.concat(fps)

        return df_predictions, eval_results, fns, fps        
    
    @staticmethod
    def draw_detect_error(video_in, gtfile_name, full_video, subset_video, frame_list, fns, fps):
        """ Draw original frames those are difficult to detect, gt annotated frames, frames with fns, and frames with fps
        Arg:
            video_in: Input video path
            gtfile_name: Ground Truth annotation json file name
            full_video: Bool to indicate whether to run the whole video, default = False
            subset_video: Number of frames to run detection on
            frame_list = List of frames with high fn and fp
            fns: False negative records in a Pandas Dataframe
            fps: False positive records in a Pandas Dataframe
        Return: 
            True once finished writing frames
        
        """

        # Capture the input video
        vid = cv2.VideoCapture(video_in)
        
        # Get total number of frames
        num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # Get GT annotations
        gt_labels = pd.read_csv('/home/ec2-user/SageMaker/0Artifact/helmet_detection/input/train_labels.csv')#.fillna(0)
        video = os.path.basename(video_in)
        print("Processing video: ",video)
        labels = gt_labels[gt_labels['video']==video]
        # if running for the whole video, then change the size of subset_video with total number of frames 
        if full_video:
            subset_video = int(num_frames)   
        for i in range(subset_video): 
            ret, frame = vid.read()
            frame_id = i+1
            if frame_id in frame_list:                
                print("Frame#: {} has high fn and fp".format(frame_id))
                ## Save each frame into a directory as .jpg - difficult to detect frames only
#                 cv2.imwrite(f"/home/ec2-user/SageMaker/0Artifact/helmet_detection/output/out_images/{i:06}.jpg", frame)

                # Get label for this frame                
                cur_label = labels[labels['frame']==frame_id] # get this frame's record
                cur_boxes = cur_label[['left','width','top','height']].values
                gt = ObjectDetector.get_gt_frame(i, cur_boxes)
                gt = gt.values.tolist()
                for frameid, class_id, x1, y1, x2, y2 in gt:
                    cv2.rectangle(frame
                                  , (x1, y1)
                                  , (x2, y2)
                                  , (255,255,0)#cyan
                                  , 2)  
                ## Save gt annotated frames into a directory as .jpg 
                cv2.imwrite(f"/home/ec2-user/SageMaker/0Artifact/helmet_detection/output/out_images/{frame_id:06}_gt.jpg", frame)

                ##### Draw fns #####
                # Get fn boxes
                fns_list = fns[fns['frame_id'] == frame_id]
                fns_list = fns_list.values.tolist()
                
               # Draw fns annotations on the frames
                for frameid, class_id, x1, y1, x2, y2 in fns_list:
                    cv2.rectangle(frame
                                  , (x1, y1)
                                  , (x2, y2)
                                  , (0,0,255)
                                  , 2)  
                ## Save fns frames into a directory as .jpg 
#                 cv2.imwrite(f"/home/ec2-user/SageMaker/0Artifact/helmet_detection/output/out_images/{i:06}_fns.jpg", frame)
                
                ##### Draw fps #####
                # Get fn boxes
                fps_list = fps[fps['frame_id'] == frame_id]
                fps_list = fps_list.values.tolist()

                for frameid, class_id, score, x1, y1, x2, y2 in fps_list:
                    cv2.rectangle(frame, 
                                  (int(x1), int(y1)), 
                                  (int(x2), int(y2)), 
                                  (255,0,0), 
                                  2)  
                ## Save fps frames into a directory as .jpg 
#                 cv2.imwrite(f"/home/ec2-user/SageMaker/0Artifact/helmet_detection/output/out_images/{i:06}_fps.jpg", frame)
                cv2.imwrite(f"/home/ec2-user/SageMaker/0Artifact/helmet_detection/output/out_images/{frame_id:06}_gt_fns_fps.jpg", frame)
                
        return True