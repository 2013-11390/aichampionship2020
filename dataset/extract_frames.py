import os
import argparse
from glob import glob
from tqdm import tqdm
import subprocess
from PIL import Image

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

def extract_frames(vid_dir, frame_dir):
    vid_fns = glob(os.path.join(vid_dir, '*.mp4'))
    for vid_fn in tqdm(vid_fns, ncols=80):
        out_fn = vid_fn.replace(vid_dir, frame_dir).replace('.mp4','/%04d.jpg')
        if not os.path.exists(os.path.dirname(out_fn)):
            os.makedirs(os.path.dirname(out_fn))
        if len(glob(os.path.join(os.path.dirname(out_fn), '*'))) != 0:
            continue
        
        status, err = extract_frame(vid_fn, out_fn)
        if not status:
            print(vid_fn, err)
            os.rmdir(os.path.dirname(out_fn))

def resize_frame(frame_dir, output_dir, size=320):
    # resize each to have smaller size as 'size'
    raw_frames = glob(os.path.join(frame_dir, '*/*.jpg'))
    for raw_fn in tqdm(raw_frames, ncols=80):
        save_fn = raw_fn.replace(frame_dir, output_dir)
        if not os.path.exists(os.path.dirname(save_fn)):
            os.makedirs(os.path.dirname(save_fn))
        if os.path.exists(save_fn):
            continue
        im = Image.open(raw_fn)
        rate = size / min(im.size)
        new_size = (int(im.size[0] * rate),int(im.size[1] * rate))
        im = im.resize(new_size, Image.ANTIALIAS)
        im.save(save_fn, quailty=95)

        
def main(vid_dir, frame_dir, output_dir, im_size):
    extract_frames(vid_dir, frame_dir)
    resize_frame(frame_dir, output_dir, im_size)
        
        
        
if __name__ == '__main__':
    description = 'Helper script for extract frames.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--vid_dir', type=str, default='./data/videos',
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('--frame_dir', type=str, default='./data/raw_frames',
                   help='Output directory where videos will be saved.')
    p.add_argument('--output_dir', type=str, default='./data/resized_rgb320',
                   help='Output directory where videos will be saved.')
    p.add_argument('--im_size', type=int, default=320,
                   help='Output directory where videos will be saved.')

    main(**vars(p.parse_args()))
