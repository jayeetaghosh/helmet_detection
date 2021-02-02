## Import the required modules
import argparse
import os, sys, time, datetime, random
import numpy as np
import pandas as pd
import cv2
import json
from PIL import Image
import subprocess
from pathlib import Path



max_distance_between_points = 30

def parse_args():
    parser = argparse.ArgumentParser(description='Object detection metrics from video and ground truth')
    parser.add_argument('video_in', type=str,
                        help='Video file to process')
    parser.add_argument('full_track', type=bool,
                        help='True - if full video  to process')
    parser.add_argument('start_ii', type=int,
                        help='Starting frame to process')
    parser.add_argument('end_ii', type=int, default=None,
                        help='End frame to process, default to last frame')
    

    args = parser.parse_args()
    return args

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)

# def get_centroid(det_box, img_height, img_width):
def get_centroid(det_box):
#     print("called get_centroid",det_box)
    
    x1 = det_box[0] #* img_width
    y1 = det_box[1] #* img_height
    x2 = det_box[2] #* img_width
    y2 = det_box[3] #* img_height
    ret = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
#     print("return", ret)
#     return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    return ret

def gt_boxes_obj_det_metrics(vid_title, frame_id, cur_boxes):
    """Get ground truth annotations in the correct format for object detection metrics
    https://github.com/rafaelpadilla/Object-Detection-Metrics
    * Create a separate ground truth text file for each image in the folder groundtruths/.
    * In these files each line should be in the format: <class_name> <left> <top> <right> <bottom>.
    * E.g. The ground truth bounding boxes of the image "2008_000034.jpg" are represented in the file "2008_000034.txt":
        bottle 6 234 45 362
        person 1 156 103 336
        person 36 111 198 416
        person 91 42 338 500
    
    Args:
        frame_id: Frame id 
        cur_boxes: Current annotation boxes 'left','width','top','height'
    Returns:
        box_ret: ground truth boxes in correct format
        class_name|Left|Top|Right|Bottom
    """

    box_out = []
#     print(type(cur_boxes))
#     print(cur_boxes)
    for box in cur_boxes:
#         print(box[0], box[2], box[1], box[3])
        box_out.append(['Helmet', box[0], box[2], box[0]+box[1], box[2]+box[3]])
        

    # Convert to dataframe before saving
    box_out = pd.DataFrame(box_out)
    gt_path = '/home/ec2-user/SageMaker/0Artifact/helmet_detection/src/helmet_detection_metric/groundtruths/'
    Path(gt_path).mkdir(parents=True, exist_ok=True)
    gt_file = gt_path + f"{frame_id:06d}.txt"
    box_out.to_csv(gt_file, sep= ' ', index = False, header=False)
    
    return None

def det_boxes_obj_det_metrics(vid_title, frame_id, cur_boxes):
    """Get ground truth annotations in the correct format for object detection metrics
    https://github.com/rafaelpadilla/Object-Detection-Metrics
    * Create a separate detection text file for each image in the folder detections/.
    * The names of the detection files must match their correspond ground truth (e.g. "detections/2008_000182.txt" represents         the detections of the ground truth: "groundtruths/2008_000182.txt").
    * In these files each line should be in the following format: <class_name> <confidence> <left> <top> <right> <bottom> (see       here * how to use it).
        E.g. "2008_000034.txt":
        bottle 0.14981 80 1 295 500  
        bus 0.12601 36 13 404 316  
        horse 0.12526 430 117 500 307  
        pottedplant 0.14585 212 78 292 118  
        tvmonitor 0.070565 388 89 500 196  

    
    Args:
        frame_id: Frame id 
        cur_boxes: Current annotation boxes 'score', 'x1','y1','x2','y2'
    Returns:
        box_ret: ground truth boxes in correct format
        class_name|confidence|Left|Top|Right|Bottom
    """

    box_out = []
#     print(cur_boxes)
    for box in cur_boxes:
        box_out.append(['Helmet', box[0], box[1], box[2], box[3], box[4]])
#         box_out.append(['Helmet', box['confidence'], box['x1'], box['y1'], 
#                            (box['x2']), (box['y2'])])

    # Convert to dataframe before saving
    box_out = pd.DataFrame(box_out)
    det_path = '/home/ec2-user/SageMaker/0Artifact/helmet_detection/src/helmet_detection_metric/detections/'
    Path(det_path).mkdir(parents=True, exist_ok=True)
    det_file = det_path + f"{frame_id:06d}.txt"
    box_out.to_csv(det_file, sep= ' ', index = False, header=False)
    
    return None



def main():
    args = parse_args()
    
    print(f"Processing video : {args.video_in}")
    
    # Get video title and output video location
    vid_title = os.path.splitext(os.path.basename(args.video_in))[0]
    # Capture the input video
    vid = cv2.VideoCapture(args.video_in)
    # Get total number of frames
    num_frames = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Set resolutions - convert them from float to integer. 
    frame_width = int(vid.get(3)) 
    frame_height = int(vid.get(4)) 
#     print(frame_width,frame_height)


#     # Get GT annotations
    gt_labels = pd.read_csv('/home/ec2-user/SageMaker/0Artifact/helmet_detection/input/train_labels.csv')#.fillna(0)
    video = os.path.basename(args.video_in)
    gt_labels = gt_labels[gt_labels['video']==video]
    print(gt_labels.shape)
#     gt_root = '/home/ec2-user/SageMaker/temp/data/task2/json_submission_files/'
#     gtfile_name = gt_root + vid_title + '-labels.json'
#     with open(gtfile_name) as f:
#         gt_labels = json.load(f)

    # Get detection annotations
    det_file = "/home/ec2-user/SageMaker/0Artifact/helmet_detection/input/" + vid_title + '.csv'
    det_labels = pd.read_csv(det_file)
    print(det_labels.shape)
#     det_name = '/home/ec2-user/SageMaker/temp/out/' + vid_title + '-detect.json'
#     with open(det_name) as f:
#         det_labels = json.load(f)
        
    # Make sure to remove previous gt, det, and results files
    gt_path = Path('detections/')
    det_path = Path('groundtruths/')
    result_path = Path('results/')
    
    try:
        [f.unlink() for f in Path(gt_path).glob("*") if f.is_file()]
    except OSError as e:
        print("Error: %s : %s" % (gt_path, e.strerror))
    try:
        [f.unlink() for f in Path(det_path).glob("*") if f.is_file()]
    except OSError as e:
        print("Error: %s : %s" % (det_path, e.strerror))
    try:
        [f.unlink() for f in Path(result_path).glob("*") if f.is_file()]
    except OSError as e:
        print("Error: %s : %s" % (result_path, e.strerror))
        
    if args.full_track:
        run_track = int(num_frames)   
    print(run_track)

    for ii in range(run_track-1): #383
            
        ret, frame = vid.read()
        
        if ii < args.start_ii:
            continue
            
        if args.end_ii:
            if ii > args.end_ii:
                break
        
        frame_id = ii+1
        print("Processing frame#: {}".format(frame_id))
        
        # Get gt label for this frame
        cur_gt_label = gt_labels[gt_labels['frame']==frame_id] # get this frame's record
#         print(cur_gt_label.columns)
        cur_gt_boxes = cur_gt_label[['left','width','top','height']].values
    
        # Get detection labels for this frame
        cur_det_label = det_labels[det_labels['frame_id']==frame_id] # get this frame's record
#         print(cur_det_label)
        cur_det_boxes = cur_det_label[['score', 'x1','y1','x2','y2']].values

        ##if there was another frame to render/process
        if ret:
#             pilimg = Image.fromarray(frame)  
            gt_boxes_obj_det_metrics(vid_title, frame_id, cur_gt_boxes)
            det_boxes_obj_det_metrics(vid_title, frame_id, cur_det_boxes)

if __name__=='__main__':
    main()

