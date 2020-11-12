import os
import argparse
from glob import glob
from tqdm import tqdm
import subprocess
from PIL import Image
from multiprocessing import Pool
from functools import partial
import time

def extract_frame(vid_fn, out_fn):
    # Get video and extract frames in 30 fps
    #ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
    command = ['ffmpeg',
               '-i', vid_fn,
               '-r', '30',
               '-q:v', '1',
               '-loglevel', 'panic',
               '"%s"' % out_fn]
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return False, err.output
    return True, 'Done'

def extract_frames(vid_fn, vid_dir, frame_dir):
    idx, vid_fn = vid_fn
    print(idx, "\r", end='')
    out_fn = vid_fn.replace(vid_dir, frame_dir).replace('.mp4','/%04d.jpg')
    if not os.path.exists(os.path.dirname(out_fn)):
        os.makedirs(os.path.dirname(out_fn))
    if len(glob(os.path.join(os.path.dirname(out_fn), '*'))) != 0:
        return        
    status, err = extract_frame(vid_fn, out_fn)
    if not status:
        print(vid_fn, err)
        os.rmdir(os.path.dirname(out_fn))
        
def main(vid_dir, frame_dir, num_worker):
    start_time = time.time()
    vid_fns = glob(os.path.join(vid_dir, '*.mp4'))
    print("total", len(vid_fns))
    vid_fns = list(enumerate(vid_fns))
    with Pool(num_worker) as p:
        p.map(partial(extract_frames, vid_dir=vid_dir, frame_dir=frame_dir), vid_fns)
    print("\n end : {}s \n".format(time.time() - start_time))
        
        
if __name__ == '__main__':
    description = 'Helper script for extract frames.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--vid_dir', type=str, default='./data_demo/videos',
                   help=('Output directory where videos are saved'))
    p.add_argument('--frame_dir', type=str, default='./data_demo/raw_frames',
                   help='Output directory where frames will be saved.')
    p.add_argument('--num_worker', type=int, default=4,
                   help='Number of workers')
    main(**vars(p.parse_args()))
