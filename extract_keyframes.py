import cv2
import os
import numpy
import json 
import time
import argparse



parser = argparse.ArgumentParser(description='keyframe extraction')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('outdir', metavar='DIR',
                    help='path to output dir')
parser.add_argument('--mode', type=str, default='train',choices=['train', 'val', 'test'], help='type of dataset')

args = parser.parse_args()

label_path = os.path.join(args.data, args.mode, 'label')

out_root = os.path.join(args.outdir, args.mode)

videolist = open(os.path.join(args.data, args.mode, 'videolist.txt'),'r')
faillist = open('keyframe_faillist.txt','w')

start = time.time()
count = 0

label = []
for line in videolist:
    label.append('sample_' + line.rstrip().split('.')[0] + '.json')
total = len(label)

for i in range(0, total):
    annotation = json.load(open(os.path.join(label_path, label[i])))
    video_id = annotation['videoid']
    shots = annotation['shots']
    for shot in shots:
        keyframe = shot['keyframe']
        try:
            output_dir = os.path.join(out_root, video_id)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file_name = os.path.join(output_dir, '%05d.jpg' % (keyframe))
            if not os.path.isfile(output_file_name):
                video_path = os.path.join(args.data, args.mode, 'video',video_id)
                video = cv2.VideoCapture(video_path)
                video.set(cv2.CAP_PROP_POS_FRAMES,keyframe)  #设置要获取的帧号
                ret, frame = video.read()
                cv2.imwrite(output_file_name,frame)
     
        except:
            print('%d %s process error' % (i, video_id))
            faillist.write(video_id+'\n')
            faillist.flush()

    if i % 100 == 0:
        print('%d / %d finished' % (i, total))
        end = time.time()
        print(end - start)
        start = end

end = time.time()
print(end - start)