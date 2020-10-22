# Train: img 890895, box_num 2303002 human_num 1507134
# Val: img 111031, box_num 290013, human_num 182270
# Test: img 235356, box_num 621969 human_num 411620

# import some common libraries
import numpy as np
import os
import argparse
import pickle
import json
import random
import cv2
import torch
from glob import glob
from tqdm import tqdm

# Setup detectron2 logger
import detectron2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# ---- Make DataSet ---------
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        with open(os.path.join(cfg.ANNO_DIR,cfg.SPLIT+'_anno.json'), 'r') as f:
            json_data = json.load(f)
        vids = []
        for elem in json_data:
            vids += elem['vids']
        self.im_paths = []
        for vid in vids:
            im_paths = glob(os.path.join(cfg.FRAME_DIR, vid + "/*.jpg"))
            im_paths.sort()
            self.im_paths += im_paths
        print(cfg.SPLIT, "num_img", len(self.im_paths))
        
    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im = cv2.imread(im_path)
        height, width = im.shape[:2]
        image = self.aug.get_transform(im).apply_image(im)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        return {"image": image, "height": height, "width": width, "im_path": im_path}
    
    def __len__(self):
        return len(self.im_paths)

def collate_fn(batch):
    return batch

def main(cfg):
    # Build Dataloader
    dataset = ImageDataset(cfg)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )

    # Bulid Model
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    model.eval()

    box_num, human_num = 0, 0
    all_boxes = {}
    for inputs in tqdm(loader, ncols=80):
        with torch.no_grad():
            result = model(inputs)

        for bidx in range(len(inputs)):
            im_path = inputs[bidx]['im_path']
            width = inputs[bidx]['width']
            height = inputs[bidx]['height']
            vid, fid = im_path.split('/')[-2:]
            fidx = int(fid.replace('.jpg',''))
            if vid not in all_boxes:
                all_boxes[vid] = {}
            if fidx not in all_boxes[vid]:
                all_boxes[vid][fidx] = []
            pred_class = result[bidx]["instances"].pred_classes.cpu().tolist() # 0: human
            pred_boxes = result[bidx]["instances"].pred_boxes.tensor.cpu().tolist()
            pred_scores = result[bidx]["instances"].scores.cpu().tolist()
            for cls, box, score in zip(pred_class, pred_boxes, pred_scores):
                all_boxes[vid][fidx].append({
                    'class': cls,
                    'box': (box[0]/width, box[1]/height, box[2]/width, box[3]/height),
                    'score': score
                })
                if cls == 0:
                    human_num += 1
            box_num += len(pred_class)

    print('box_num', box_num, 'human_num', human_num)

    with open(os.path.join(cfg.ANNO_DIR, cfg.SPLIT+'_pred_bboxes.pkl') ,'wb') as f:
        pickle.dump(all_boxes, f)

    
if __name__ == '__main__':
    description = 'Helper script for extract frames.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--anno_dir', type=str, default='./data',
                   help='Annotation dir path')
    p.add_argument('--frame_dir', type=str, default='./data/raw_frames',
                   help='Frames dir path')
    p.add_argument('--split', type=str, default='test',
                   help='Data split [train, val, test]')
    args = p.parse_args()
    
    # Build Config
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    cfg.MODEL.MASK_ON = False
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.ANNO_DIR = args.anno_dir
    cfg.FRAME_DIR = args.frame_dir
    cfg.SPLIT = args.split
    main(cfg)
