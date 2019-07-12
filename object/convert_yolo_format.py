import os
import json
import cv2
import time
import argparse

parser = argparse.ArgumentParser(description='convert object label')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('keyframe_dir', metavar='DIR',
                    help='path to frame dir')
parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'])

args = parser.parse_args()

root = os.path.join(args.data, args.mode) 

video_list = open(os.path.join(root, 'videolist.txt'),'r')

output_list = open('%s_list.txt' % args.mode,'w')

obj_name = open('objects_en.txt','r')
obj_list = [line.rstrip() for line in obj_name]


train_category = []

video_count = 0
start = time.time()
key_frame_count = 0
output_folder = '%s_label' % args.mode
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for vid in video_list:

    label = json.load(open(os.path.join(root, 'label', 'sample_' + vid.rstrip().split('.')[0] + '.json'), 'r'))
    init = False
    for shot in label['shots']:
        keyframe = shot['keyframe']
        image_path = os.path.join(args.keyframe_dir, args.mode, vid.split('.')[0], '%05d.jpg'% keyframe)

        if os.path.isfile(image_path):
            output_list.write(image_path)
            output_list.write('\n')
            
            if args.mode != 'test':
                if init == False:
                    img = cv2.imread(image_path) 
                    height, width, _ = img.shape
                    init = True    
                    dw = 1. / width
                    dh = 1. /height

                outfile_folder = os.path.join(output_folder,  vid.split('.')[0])
                if not os.path.exists(outfile_folder):
                    os.makedirs(outfile_folder)
                outfile_name = os.path.join(outfile_folder, '%05d.txt'% keyframe)
                write = False
                if not os.path.isfile(outfile_name):
                    out_file = open(outfile_name, 'w')
                    write = True

                for target in shot['targets'] :
                    if target['category'] == 0:
                        xmin = target['bbox']['x']
                        xmax = xmin + target['bbox']['width']
                        ymin = target['bbox']['y']
                        ymax = ymin + target['bbox']['height']
                        x = (xmin + xmax) /2.0
                        y = (ymin + ymax) /2.0
                        w = xmax - xmin
                        h = ymax - ymin
                        x = x * dw
                        w = w * dw
                        y = y * dh
                        h = h * dh
                
                        cls_id = target['tag']
                        if write == True:
                            out_file.write(str(cls_id) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + '\n')
                out_file.close()
            else:
                outfile_folder = os.path.join(output_folder,  video_id)
                if not os.path.exists(outfile_folder):
                    os.makedirs(outfile_folder)
                outfile_name = os.path.join(outfile_folder, '%05d.txt'% keyframe)
                if not os.path.isfile(outfile_name):
                    out_file = open(outfile_name, 'w')
                    out_file.write(str(0) + " " + str(0) + " " + str(0) + " " + str(0) + " " + str(0) + '\n')
                out_file.close()
    if video_count % 100 ==0:
        print(video_count)
        end = time.time()
        print(end-start)
        start = end
    video_count += 1

