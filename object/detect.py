from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import json

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_list", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode


    dataset = ListDataset(opt.image_list, img_size=opt.img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, collate_fn=dataset.collate_fn
    )
    
    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs_path = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (file_path, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
       
       
        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, opt.conf_thres, opt.nms_thres)
        
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        
        for (idx, path) in enumerate(file_path):
            
            img = np.array(Image.open(path))
            detections = outputs[idx]
  

            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, opt.img_size, img.shape[:2])

                #result_file_name = path.replace(args.keyframes,'./results/').replace('jpg','json')
                result_folder = './object_results/'
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)
                videoid = path.split('/')[-2]
                keyframe = path.split('/')[-1].split('.')[0]
                result_file_name = os.path.join(result_folder, videoid + '_%s.json' % keyframe)
                res = []
                best_score = 0
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                    #print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                    if cls_conf.item()>best_score:
                        x = x1.item()
                        y = y1.item()
                        x_max = x2.item()
                        y_max = y2.item()
                        if x < 0:
                            x = 0
                        if y < 0:
                            y = 0
                        if x > h:
                            x = h
                        if y > h:
                            y = h
                        box_w = x_max - x
                        box_h = y_max - y
                        bbox = {'x':x, 'y':y, 'width':box_w, 'height':box_h}
                        
                            #print(bbox)
                        output = {'tag':int(cls_pred.item()), 'score':cls_conf.item(),'bbox':bbox,'image_id':result_file_name.split('/')[-1]}
                        best_score = cls_conf.item()
                        #res.append(output)
                json.dump(output, open(result_file_name,'w'), indent=4)
                #print(outputs[idx])

    