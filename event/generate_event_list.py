import os
import json
import argparse

parser = argparse.ArgumentParser(description='generate event list')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('keyframe_dir', metavar='DIR',
                    help='path to frame dir')
parser.add_argument('--mode', type=str, default='train',choices=['train', 'val', 'test'])

args = parser.parse_args()

event_list_file = open('event_list.txt','r')
event_list = [line.rstrip() for line in event_list_file]

root = os.path.join(args.keyframe_dir, args.mode) 

if not os.path.exists('./split'):
        os.makedirs('./split')
output_list = open('./split/%s_list.txt' % args.mode,'w')

total = 0

for (idx, video) in enumerate(os.listdir(root)):
    
    duration = len(os.listdir(os.path.join(root, video)))
    if duration > 0:
        
        video_path = os.path.join(root, video)
        if args.mode == 'test' :
                output_list.write('%s %d\n' % (video_path, duration))
        else:
                label = os.path.join(args.data, args.mode, 'label','sample_' + video + '.json')
                annotation = json.load(open(label,'r'))

                #video_id = annotation['videoid']

                video_class = event_list.index(annotation['videoclass'])

                output_list.write('%s %d %d\n' % (video_path, duration, video_class))
        total += 1
print(total)
