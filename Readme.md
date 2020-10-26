# Video Classification

### Install
python 3, torch 1.4

- Need ffmpeg for processing video
```
sudo apt install ffmpeg
```

- Clone repository and install python packages
```
git clone https://github.com/2013-11390/aichampionship2020.git
cd aichampionship2020
pip install -r requirement.txt
pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

- Install Detectron2
```
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```
OR you can install pre-built Detectron2 for your CUDA and torch version in linux [here](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

### Prepare Dataset

- Download KDX_train.csv file and move it to PATH_TO_DATA_CSV

- Download videos and trim
This will download video from youtube then save trimmed video clips in PATH_TO_VID.
```
python dataset/download_vid.py --input_csv PATH_TO_DATA_CSV --output_dir PATH_TO_VID
```

- Extract frames from videos
This will extract frames from video clips into PATH_TO_FRAMES (30fps). 
Then resize images into minimum side length as 255.
```
python dataset/extract_frames.py --vid PATH_TO_VID --output_dir PATH_TO_FRAME
```

- Make annotation file, We random split dataset into train, val, test as 70%, 10%, 20% respectively.
This will create train_anno.json, val_anno.json for training and validation, and test_gt.csv file for testing
```
python dataset/make_annotation.py --output_dir PATH_TO_ANNO
```

- Extract object bounding box use detectron2
```
python dataset/object_detection.py --ann_dir PATH_TO_ANNO --frame_dir PATH_TO_FRAME
```



### Training
- Check configs/SLOWFAST_8x8_R50.yaml
  - Change DATA.PATH_TO_DATA_DIR to PATH_TO_ANNOTATION
  - Change DATA.PATH_PREFIX to PATH_TO_FRAMES

- Prepare Kinetics pretrained checkpoint

- Run train.py
```
python train.py --cfg YAML_PATH --tag TAG
```
The checkpoint will save in logdir/TAG

### Evaluation
- Run test.py (use TAG for testing model saved in logdir/TAG)
```
python test.py --cfg TEST_YAML_PATH --tag TAG --test_tag TEST_TAG
```
