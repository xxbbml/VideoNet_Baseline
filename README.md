# VideoNet_Baseline
Baseline method for VideoNet Competition

# Data processing

## Requirements:

ffmpeg (>=2.8.15)

ffprob (>=2.8.15)

python+OpenCV 3.4.1

### 1. Download datasets
 Download datasets from the VideoNet website and put in *root* folder. 
### 2. Download train, val, and test videos 

Use download_video.py from each subset.
### 3. Extract frames and keyframes from videos. 
We extract one frame every 60 frames for efficiency. You can adjust the step with --step.
   
   ```
   python3 data_processing.py root output_dir --mode {train, val, test} --step 60
```
### 4. You can also extract keyframes only using the code extract_keyframes.py
    
    python3 extract_keyframe.py root output_dir --mode {train, val, test} 
    
### 5. All the model used in baseline and submission sample can be downloaded from [Baidu Yun](https://pan.baidu.com/s/1HXL_zto755jBrbeqqdT0fg) (Code: 78yn) or [Google Drive](https://drive.google.com/open?id=1jZoDtTUFmDcGHNeZV2SfMcEMkBb5jwIH).

# Event
Our baseline uses Temporal Segment Network to predict the event class of videos. More details can refer to the paper and original repo.

[Temporal Segment Network](https://github.com/yjxiong/tsn-pytorch)

## Requirements:

PyTorch >= 0.4.1

We only train our model on RGB images. The model is finetuned from Kinetics.

### 1. Generate event list:
 ```
python3 generate_event_list.py root keyframe_dir --mode {train, val, test}
 ```
### 2. Inference:
Download *VideoNet_bninception__rgb_model_best.pth.tar* and put it under the event folder.

 ```
python3 test_models.py VideoNet RGB ./split/test_list.txt VideoNet_bninception__rgb_model_best.pth.tar --arch BNInception --save_scores event_results.json --gpus 0 1 2 3 -j 4
 ```

The results will be saved in the event_results.json.

### 3. Training:
Download *kinetics_tsn_rgb.pth.tar* and put it under ./tf_model_zoo.

 ```
python3 main.py VideoNet RGB ./split/train_list.txt ./split/val_list.txt --arch BNInception --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 128 -j 8 --dropout 0.8 --snapshot_pref VideoNet_bninception_  --gpus 0 1 2 3
 ```

# Scene
Our baseline uses PlaceCNN of ResNet-50 to predict the scene classes of keyframes. More details can refer to the paper and original repo.

[PlaceCNN](https://github.com/CSAILVision/places365)

## Requirements:

PyTorch >= 0.4.1

We finetune our model from Place365.
### 1. Generate scene list:
 ```
python3 generate_scene_list.py root keyframe_dir --mode {train, val, test}
 ```

### 2. Inference:

Download *resnet50_best.pth.tar* and put it under the scene folder.

 ```
python run_placesCNN.py scene_label_list_test.txt weights --arch resnet50 
 ```

The results will be saved in the ./scene_results folder.

### 3. Training:
Download *resnet50_places365.pth.tar* and put it under the scene folder.
 ```
python train_placesCNN.py -a resnet50 --train scene_label_list_train.txt --val scene_label_list_val.txt
 ```

# Object

Our baseline uses YOLOv3 to predict the object classes of keyframes. More details can refer to the paper and original repo. 

[PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

## Requirements:

PyTorch = 1.1.0

We finetune our model from Darknet.

### 1. Generate object list:
 ```
python3 convert_yolo_format.py root keyframe_dir --mode {train, val, test}
 ```

### 2. Inference:

Download *yolov3_ckpt_6.pth* and put it under the object folder.
 ```
python3 detect.py --weights_path checkpoints/yolov3_ckpt_6.pth --model_def config/yolov3-custom.cfg --image_list test_list.txt --class_path objects_en.txt
 ```

The results will be saved in the ./object_results folder with one instance each image.


### 3. Training:
As for training, you may need to modify the path in config/custom.data.
You also need to modify the 65 line in utils/datasets.py with *keyframe dir*.
```
path.replace(train_frame_folder, './train_label/').replace(val_frame_folder, './val_label/').replace(".png", ".txt").replace(".jpg", ".txt")

 ```

And Download *darknet53.conv.74* and put it under the ./weights.

 ```
python3 train.py --data_config config/custom.data  --pretrained_weights weights/darknet53.conv.74 --model_def config/yolov3-custom.cfg
 ```

# Submission

Run generate_submission.py to generate submission as baseline.json (The submission sample file can be downloaded from Baidu Yun or Google drive). The two threshold will filter the outputs with low confidence since some images has no scenes or objects in our label lists.

 ```
python3 generate_submission.py root --scene_thres --object_thres
 ```

 # Results

 The results of our baseline:
 
|        | Event | Object | Scene | Total Score|
| ---------- | --- | --- | --- | --- |
| Validation |  77.30 | 25.58| 55.37 | 0.4784 |
| Test       | 78.17  | 23.42 | 55.29| 0.4712 |

The object_thres = 0.9 and scene_thres = 0.7.
