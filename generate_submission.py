import json
import os
import random
import argparse

parser = argparse.ArgumentParser(description='generate submission')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--scene_thres', type=float, default=0.7)
parser.add_argument('--object_thres', type=float, default=0.7)
parser.add_argument('--mode', type=str, default='test', choices=['val', 'test'], help='type of dataset')

args = parser.parse_args()

test_list = open(os.path.join(args.data,args.mode,'videolist.txt'),'r')

event_list = json.load(open('./event/event_results.json','r'))

gt = []

count = 0

for line in test_list:
    
    video = line.rstrip()

    label = os.path.join(args.data, args.mode,'label','sample_' + video.split('.')[0] + '.json')

    test_annotation = json.load(open(label,'r'))
    result = {}
    result['videoid'] = test_annotation['videoid']
    if test_annotation['videoid'].split('.')[0] in event_list:
        result['videoclass'] = event_list[test_annotation['videoid'].split('.')[0]]
    else:
        result['videoclass'] = 0
    result['shots'] = test_annotation['shots']
    
    for shot in result['shots']:
        shot['targets'] = []
        tar = {}
        keyframe = shot['keyframe']

        
        scene_path = os.path.join('./scene/scene_results/', video.split('.')[0] + '_%05d.json' % keyframe)
        if os.path.isfile(scene_path):
            scene = json.load(open(scene_path, 'r'))
            if scene['prob'] > args.scene_thres:
                tar['category'] = 1
                tar['tag'] = scene['prediction']
                shot['targets'].append(tar)
       
        tar = {}

        obj_path = os.path.join('./object/object_results/', video.split('.')[0] + '_%05d.json' % keyframe)

        if os.path.isfile(obj_path):
            obj = json.load(open(obj_path, 'r'))
            if obj['score'] > args.object_thres:
                tar['category'] = 0
                tar['bbox'] = obj['bbox']
                tar['tag'] = obj['tag']
                shot['targets'].append(tar)

    gt.append(result)
    
json.dump(gt, open('baseline.json','w'), indent=4)   


