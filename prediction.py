
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def sorting_index(data):
    df_duplicate = [i for i in data]
    list_idx = []
    new_list = []
    while len(df_duplicate) > 0:
        #parse data
        min_xy = [x+y for x,y,w,h in data]
        x_data = [x for x,y,w,h in data]
        y_data = [y for  x,y,w,h in data]
        h_data = [h for  x,y,w,h in data]
        #cek data di min_xy sudah ada di list_idx atau belum
        list_val = [(idx, val) for idx, val in enumerate(min_xy) if idx not in list_idx]
        if len(list_val) >0:
            #Mencari nilai terkecil dari list_val
            min_idx_val = min([val for idx, val in list_val])
            start_row = int([idx for idx,val in list_val if val==min_idx_val][0])
            list_idx.append(start_row)
            new_list.append(data[start_row])
            del df_duplicate[0]

            y_start = y_data[start_row] - int((h_data[start_row])/2)
            y_range = y_data[start_row] + int((h_data[start_row])/2)

            in_range = [(idx, val) for idx,val in enumerate(y_data) if val in range(y_start, y_range) if idx != start_row]
            for i in range(0, len(in_range)):
                min_val = [(x_data[idx], y_data[idx]) for idx, val in in_range if idx not in list_idx]
                if len(min_val) >0:
                    min_val = min(min_val)
                    next_column = int([idx for idx, val in in_range if (x_data[idx], y_data[idx]) == min_val][0])
                    list_idx.append(next_column)
                    new_list.append(data[next_column])
                    del df_duplicate[0] 
                else:
                    pass     
        else:
            pass
    return new_list

@torch.no_grad()
def detect(weights,  # model.pt path(s)
        source,  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        # obj_double="cage_wheel_track",
        ):
    # source = str(source)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        
        pred = model(im, augment=augment, visualize=visualize)
    # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                # names_class = det[:, -1].unique()
                xywh = []
                for *xyxy, conf, cls in reversed(det):
                    xywh.append((int(xyxy[0]+ (xyxy[2]-xyxy[0])/2), int(xyxy[1]+ (xyxy[3]-xyxy[1])/2),int(xyxy[2]-xyxy[0]),int(xyxy[3]-xyxy[1])))
            else:
                xywh = None
                class_list = None

    if xywh is not None:
        sorted_idx = sorting_index(xywh)
        df = (pd.DataFrame(sorted_idx)).rename(columns={0:'xcenter', 1: 'ycenter', 2:'width', 3:'height'}) # create df from data, without count column
        df['idx']= df.index+1
        radius = [min(data[2], data[3]) // 2.5 for data in df.values]
        df['radius'] = radius
        return df
    else:
        column_names = ["xcenter", "ycenter", "width", "height"]
        df = (pd.DataFrame(columns = column_names))
        return df
    
def get_img_size(img_path):
    im = cv2.imread(img_path)
    im_sz = im.shape
    return im_sz[1], im_sz[0] # width & height