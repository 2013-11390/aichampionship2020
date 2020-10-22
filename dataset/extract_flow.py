import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
TVL1 = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1,epsilon=0.05,warps=1)

def cal_for_frames(video_path):
    
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()
    #print(video_path, len(frames))

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    #prev = cv2.resize(prev,(224,224))
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        #curr = cv2.resize(curr,(224,224))
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow

def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""    
    flow = TVL1.calc(prev, curr, None)
    flow = np.clip(flow, -20,20) #default values are +20 and -20
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow

def save_flow(video_flows, flow_path):
    #u is the first channel and v is the second channel.
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, "u_{:04d}.jpg".format(i)),
                    flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, "v_{:04d}.jpg".format(i)),
                    flow[:, :, 1])


frame_dir='./data/resized_rgb320' 
flow_dir='./data/tvl1' 
frame_vid_paths = glob(os.path.join(frame_dir, '*'))

for frame_path in tqdm(frame_vid_paths):
    flow_path = frame_path.replace(frame_dir, flow_dir)
    if not os.path.exists(flow_path):
        os.makedirs(flow_path)
    #Convert your video to frames and save in to activity_rgb folder
    video_flows = cal_for_frames(frame_path)
    save_flow(video_flows, flow_path)
