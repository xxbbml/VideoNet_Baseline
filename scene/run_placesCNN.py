# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
from PIL import Image
import time
import json
import argparse

parser = argparse.ArgumentParser(
    description="Scene image testing")

parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str, default="resnet50_best.pth.tar")
parser.add_argument('--arch', type=str, default="resnet50")

args = parser.parse_args()

# th architecture to use
arch = args.arch
class_to_idx = json.load(open('class_to_idx.json','r'))
idx_to_class = {v : k for k, v in class_to_idx.items()}

# load the pre-trained weights
model_file = args.weights
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=205)
#checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
checkpoint = torch.load(model_file)

state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}

model.load_state_dict(state_dict)

model.cuda()
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'scene_en.txt'

classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip())
classes = tuple(classes)

# load the test image

frame_list = open(args.test_list,'r')

count = 0
start = time.time()

if not os.path.exists('scene_results'):
    os.makedirs('scene_results')

for img in frame_list:
    
    videoid = img.rstrip().split('/')[-2]
    img_name = img.rstrip().split('/')[-1].split('.')[0]
    result_name = 'scene_results/%s_%s.json' % (videoid.split('.')[0], img_name)
    if not os.path.isfile(result_name):

        try:
            image = Image.open(img.rstrip())
            input_img = V(centre_crop(image).unsqueeze(0))

            input_img = input_img.cuda(async=True)

            logit = model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            
            #print('{} prediction on {}'.format(arch,img_name))
            # output the prediction
            pred_label = idx[0].item()
            pred_label = int(idx_to_class[pred_label])
            prediction = {'videoid':videoid, 'keyframe':img.rstrip().split('/')[-1].split('.')[0][-5:], 'prediction':pred_label,'class':classes[pred_label], 'prob':probs[0].item()}
            result_file = open(result_name,'w')

            json.dump(prediction, result_file, indent=4)
            result_file.close()
        except:
            print('Process %s frame in %s video error' % (img_name, videoid))
    count += 1
    if count % 100 == 0:
        print('%d image finished' % (count))
        end = time.time() 
        print(end-start)
        start = end