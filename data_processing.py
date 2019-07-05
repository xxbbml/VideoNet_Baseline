import os
import json
from PIL import Image
import subprocess as sp
import argparse

def get_params(filepath):
    '''
    获取目标文件的视频流信息
    get_params(filepath)->info

    参数列表：
        filepath (str) 视频路径

    返回值：
        info (dict) 视频流信息
    '''
    command = ['ffprobe', '-print_format', 'json',
               '-show_streams', '-i', str(filepath)]
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    infos = pipe.stdout.read()
    jsondat = json.loads(str(infos, encoding='utf8'))
    for stream in jsondat['streams']:
        width = stream.get('width')
        height = stream.get('height')
        duration = stream.get('duration')
        frame_rate = stream.get('avg_frame_rate')
        if (width and height and frame_rate) != None:
            ret = stream
            break
    return ret
    

parser = argparse.ArgumentParser(description='data processing')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('outdir', metavar='DIR',
                    help='path to output dir')
parser.add_argument('--mode', type=str, default='train',choices=['train', 'val', 'test'], help='type of dataset')
parser.add_argument('--step', type=int, default=60)

args = parser.parse_args()

video_list = open(os.path.join(args.data, args.mode, 'videolist.txt'),'r') 
faillist = open('faillist.txt','w')

for (idx, line) in enumerate(video_list):

    video = line.rstrip()
    label = os.path.join(args.data, args.mode, 'label', 'sample_' + video.split('.')[0] + '.json')
    annotation = json.load(open(label,'r'))

    video_id = video
    output_dir = os.path.join(args.outdir, args.mode, video_id.split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        keyframe_list = []
        for shot in annotation['shots']:
            keyframe_list.append(shot['keyframe'])

        filepath = os.path.join(args.data,args.mode,'video', video_id)

        try:
            params = get_params(filepath)
            bytes_per_frame = params['width'] * params['height'] * 3

            command = [ 'ffmpeg',
                    '-i', filepath,
                    '-f', 'image2pipe',
                    '-pix_fmt', 'rgb24',
                    '-vcodec', 'rawvideo', '-']
            pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=bytes_per_frame * 3)

            frame_count = 0
            while True:
                try:
                    raw = pipe.stdout.read(bytes_per_frame)
                    if frame_count % args.step == 0 or frame_count in keyframe_list:
                        frame = Image.frombytes(
                        'RGB', (params['width'], params['height']), raw)
                        pipe.stdout.flush()
                        output_file_name = os.path.join(output_dir, '%05d.jpg' % frame_count )
                        if not os.path.isfile(output_file_name) :
                            frame.save(output_file_name)
                    frame_count += 1
                except ValueError as inadequate_data:
                    # 解码结束后缓冲区无数据，但数据不足时Image.frombytes抛出ValueError。
                    print(inadequate_data)
                    pipe.terminate()
                    break
        except:
            print('Fail to extract frames from %s' % video)
            faillist.write(video)
            faillist.write('\n')
faillist.close()