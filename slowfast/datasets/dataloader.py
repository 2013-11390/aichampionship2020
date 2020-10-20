import itertools
import json
import torch
from torch.utils.data._utils.collate import default_collate
import random
from glob import glob
import os
import numpy as np

from . import utils

class KDX(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        """
        Load KDX data (frame paths, labels) to a given Dataset object.
        Args:
            cfg (CfgNode): configs.
            split (string): Options includes `train`, `val`, or `test` mode.
                For the train mode, the data loader will take data
                from the train set, and randomlly sample one clip per video.
                For the val mode, the data loader will take data
                from the val set, and sample one clip in the center per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
        """
        assert (split in ("train","val","test")), "Split '{}' not supported".format(split)
        self.split = split
        self.cfg = cfg # cfg.DATA.PATH_TO_DATA_DIR, self.cfg.DATA.PATH_PREFIX, cfg.DATA.SAMPLING, 
        
        # Load annotation file [{label: str, vid: [str]}]
        with open(os.path.join(cfg.DATA.PATH_TO_DATA_DIR, split+'_anno.json'), 'r') as f:
            json_data = json.load(f)
        
        # Get label info
        self.label2idx, self.idx2label = {}, []
        for idx, elem in enumerate(json_data):
            label = elem['label']
            self.label2idx[label] = idx
            self.idx2label.append(label)
        
        # Process data
        self.data = []
        self.sampling = False
        if self.split == 'train' and self.cfg.DATA.SAMPLING > 0:
            self.sampling = True
            self.train_data = json_data
            self.data = list(range(len(json_data))) * self.cfg.DATA.SAMPLING # 1epoch: 25 x 200 samples
        elif self.split == 'test':
            for elem in json_data:
                label = elem['label']
                for vid in elem['vids']:
                    for e_idx in range(self.cfg.TEST.NUM_ENSEMBLE_VIEWS * self.cfg.TEST.NUM_SPATIAL_CROPS):
                        # e_idx: ensemble index
                        self.data.append((label, vid, e_idx)) 
        else:
            for elem in json_data:
                label = elem['label']
                for vid in elem['vids']:
                    self.data.append((label, vid))
    
    def get_frames(self, vid, e_idx=-1):
        # e_idx = -1: random sampling, -2: centor crop, >=0: fixed location
        frame_paths = glob(os.path.join(self.cfg.DATA.PATH_PREFIX, vid, '*.jpg'))
        frame_paths.sort()
        video_length = len(frame_paths)
        
        # Perform temporal sampling
        num_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = self.cfg.DATA.SAMPLING_RATE
        clip_length = (num_frames - 1) * sampling_rate + 1
        if e_idx == -1:
            if clip_length > video_length:
                start = random.randint(video_length - clip_length, 0)
            else:
                start = random.randint(0, video_length - clip_length)
        elif e_idx == -2:
            gap = float(max(video_length - clip_length, 0)) / 2
            start = int(round(gap))
        else:
            temporal_sample_index = e_idx // self.cfg.TEST.NUM_SPATIAL_CROPS
            gap = float(max(video_length - clip_length, 0)) / (
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1
            )
            start = int(round(gap * temporal_sample_index))
        seq = [
            max(min(start + i * sampling_rate, video_length - 1), 0)
            for i in range(num_frames)
        ]
        
        # Load images  num_frames size list
        frames = torch.as_tensor(
            utils.retry_load_images(
                [frame_paths[frame] for frame in seq],
                retry=2,
            )
        )
        
        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        
        # Perform spatial sampling
        if e_idx == -1:
            spatial_sample_index = -1
            min_scale, max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif e_idx == -2:
            spatial_sample_index = 1
            min_scale, max_scale, crop_size = 256, 256, self.cfg.DATA.TRAIN_CROP_SIZE
        else:
            spatial_sample_index = e_idx % self.cfg.TEST.NUM_SPATIAL_CROPS
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3

        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP, # True
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE, # False
        )
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames
        
    def __getitem__(self, idx):
        e_idx = -1 if self.split == 'train' else -2
        if self.sampling:
            label_elems = self.train_data[self.data[idx]]
            label = label_elems['label']
            vid = random.choice(label_elems['vids'])
        elif self.split == 'test':
            label, vid, e_idx = self.data[idx]
        else:
            label, vid = self.data[idx]
            
        # Extract frame from vid
        frames = self.get_frames(vid, e_idx)
        # Get label index
        label_idx = self.label2idx[label]
        
        return frames, label_idx, vid, {}

    def __len__(self):
        return len(self.data)
        
        
def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    labels = torch.tensor(np.concatenate(labels, axis=0)).float()

    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == "ori_boxes":
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(
                list(itertools.chain(*data))
            ).view(-1, 2)
        else:
            collated_extra_data[key] = default_collate(data)

    return inputs, labels, video_idx, collated_extra_data

def construct_loader(cfg, split):
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
        
    dataset = KDX(cfg, split)
    
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )

    return loader

if __name__ == '__main__':
    print("hi")