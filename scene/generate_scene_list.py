import os
import numpy
import json 
import time
import shutil
import cv2
import argparse

parser = argparse.ArgumentParser(description='generate scene list')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('keyframe_dir', metavar='DIR',
                    help='path to frame dir')
parser.add_argument('--mode', type=str, default='train',choices=['train', 'val', 'test'])

args = parser.parse_args()

label_path = os.path.join(args.data, args.mode, 'label')
video_path = os.path.join(args.keyframe_dir, args.mode)

out_label = open('scene_label_list_%s.txt' % args.mode,'w')
faillist = open('faillist.txt','w')

start = time.time()
for video in os.listdir(video_path):
    frame_path = os.path.join(video_path, video)
    label_name = 'sample_' + video.split('.')[0] + '.json'
    annotation = json.load(open(os.path.join(label_path, label_name)))
    
    shots = annotation['shots']
    for shot in shots:
        keyframe = shot['keyframe']
        if args.mode == 'test':
            frame_path = os.path.join(video_path, video, '%05d.jpg' % (keyframe))
            if not os.path.isfile(frame_path):
                print('cant find image %d in %s ' % (keyframe, video))
                faillist.write('%s %d\n' % (video, keyframe))
            else:
                out_label.write('%s\n' % frame_path)
        else:
            for target in shot['targets']:
                if target['category'] == 1:
                    frame_path = os.path.join(video_path, video, '%05d.jpg' % (keyframe))
                    if not os.path.isfile(frame_path):
                        print('cant find image %d in %s ' % (keyframe, video))
                        faillist.write('%s %d\n' % (video, keyframe))

                    else:
                        out_label.write('%s %d\n' % (frame_path, target['tag']))
        
           