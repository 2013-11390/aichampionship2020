import argparse
import glob
import os
import pandas as pd
import json
import subprocess
from tqdm import tqdm
from collections import defaultdict
import uuid

def download_clip(video_id, times,
                  output_dir='./data/videos',
                  tmp_dir='./data/tmp',
                  url_base='https://www.youtube.com/watch?v='):
    """Download a video from youtube if exists and is not blocked.
    arguments:
    ---------
    video_id: str
        Unique YouTube video identifier (11 characters)
    times: list [(start_time, end_time)]
        Start and end time in seconds from where the video will be trimmed.
    """
    
    # Defensive argument checking.ㅃ
    assert isinstance(video_id, str), 'video_identifier must be string'
    assert len(video_id) == 11, 'video_identifier must have length 11'

    # Construct command line for getting the direct video link.
    tmp_filename = os.path.join(tmp_dir, video_id+'.mp4')
    error = 0
    while True:
        # Download Video
        command = ['youtube-dl',
                   '--quiet', '--no-warnings',
                   '-f', 'mp4',
                   '-o', '"%s"' % tmp_filename,
                   '"%s"' % (url_base + video_id)]
        command = ' '.join(command)
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
            break
        except subprocess.CalledProcessError as err:
            if error < 3:
                error += 1
            else:
                return False, err.output

    # Construct command to trim the videos (ffmpeg required).
    status, err_message = True, ''
    for start_time, end_time in times:
        output_filename = os.path.join(output_dir, '{}_{}_{}.mp4'.format(video_id,start_time,end_time))
        if os.path.exists(output_filename):
            continue
        command = ['ffmpeg',
                   '-i', '"%s"' % tmp_filename,
                   '-ss', str(start_time),
                   '-t', str(end_time - start_time),    
                   '-c:v', 'libx264', '-c:a', 'copy',
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '"%s"' % output_filename]
        command = ' '.join(command)
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            status = False
            err_message = err.output
    if not status:
        return False, err_message
    
    # Check if the video was successfully saved.
    status = os.path.exists(output_filename)
    os.remove(tmp_filename)
    return status, 'Downloaded'

def create_video_folders(output_dir, tmp_dir):
    """Creates a directory for each label name in the dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

def process_files(input_csv):
    '''
    args
        input_csv: str, input csv filename
    return
        dataset: {youtube_id: [(start_time, end_time), ]}
    '''
    
    data_csv = pd.read_csv(input_csv)
    data_npy = data_csv.to_numpy()
    
    dataset = defaultdict(list)
    for elem in data_npy:
        dataset[elem[2]].append(
            (elem[3], elem[4])
        )
    
    return dataset
        

def main(input_csv, output_dir, tmp_dir, num_device, num, **kwargs):
    # dataset: 'video-id', 'start-time', 'end-time', 'label-name'
    dataset = process_files(input_csv)
    video_keys = sorted(list(dataset.keys()))
    
    length = len(video_keys) // num_device
    if num == num_device-1:
        video_keys = video_keys[num*length:]
    else:
        video_keys = video_keys[num*length:(num+1)*length]
    print(num)
    
    create_video_folders(output_dir, tmp_dir)
    
    for key in tqdm(video_keys, ncols=80):
        video_id = key
        status, err = download_clip(video_id, dataset[key], output_dir)
        if not status:
            print(video_id, err)

            
def re_download(input_csv, vid_dir, tmp_dir, url_base='https://www.youtube.com/watch?v='):
    dataset = process_files(input_csv)
        
    # Check download
    not_downloaded = []
    for video_id in dataset:
        for start_time, end_time in dataset[video_id]:
            vid = '{}_{}_{}'.format(video_id, start_time, end_time)
            v_path = os.path.join(vid_dir, vid+'.mp4')
            if not os.path.exists(v_path):
                not_downloaded.append(vid)
    print('Try to download', len(not_downloaded))
    
    for vid in tqdm(not_downloaded, ncols=80):
        video_id, start_time, end_time = vid.rsplit('_', 2)
        tmp_filename= os.path.join(tmp_dir, video_id)
        if len(glob.glob(tmp_filename+'*')) == 0:
            command = ['youtube-dl',
                       '--quiet', '--no-warnings',
                       '-o', '"%s"' % tmp_filename,
                       '"%s"' % (url_base + video_id)]
            command = ' '.join(command)
            try:
                output = subprocess.check_output(command, shell=True,
                                                 stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as err:
                print(video_id, err)
                continue
        output_filename = os.path.join(vid_dir, vid+'.mp4')
        tmp_filename = glob.glob(tmp_filename+'*')[0]
        command = ['ffmpeg',
                   '-i', '"%s"' % tmp_filename,
                   '-ss', start_time,
                   '-t', str(int(end_time) - int(start_time)),    
                   '-c:v', 'libx264', '-c:a', 'copy',
                   '-threads', '1',
                   '-loglevel', 'panic',
                   '"%s"' % output_filename]
        command = ' '.join(command)
        try:
            output = subprocess.check_output(command, shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            status = False
            print('ffmpeg', video_id, err)
        
if __name__ == '__main__':
    description = 'Helper script for downloading and trimming videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--input_csv', type=str, default='./data/KDX_train.csv',
                   help=('CSV file containing the following format: '
                         'label,youtube_id,time_start,time_end'))
    p.add_argument('--output_dir', type=str, default='./data/videos',
                   help='Output directory where videos will be saved.')
    p.add_argument('-t', '--tmp_dir', type=str, default='./data/tmp')
    p.add_argument('--num_device', type=int, default=4)
    p.add_argument('--num', type=int)
    p.add_argument('--re_download', action='store_true')
    
    args = p.parse_args()
    if args.re_download:
        re_download(args.input_csv, args.output_dir, args.tmp_dir)
    else:
        main(**vars(args))
    