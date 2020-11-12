import os
import argparse
import glob
from tqdm import tqdm
import subprocess
from PIL import Image
from multiprocessing import Pool
from functools import partial
import pandas as pd
import time


def split_vid(elem, tmp_dir, out_dir):
    idx, elem = elem
    #print(idx, "\r", end='')
    #print(idx, elem)
    yid, title = elem['yid'], elem['title']
    title = title.replace('[', '[[]').replace('?', '*').replace('/', '*')
    start_time, end_time = elem['start_time'], elem['end_time']
    tmp_fn_cands = glob.glob(os.path.join(tmp_dir, title +'*'))
    if len(tmp_fn_cands) == 0:
        #import pudb;pudb.set_trace()
        print("Vid not exist", yid, title)
        return #False, "Vid not exist"
    tmp_fn = tmp_fn_cands[0]
    out_fn = os.path.join(out_dir, '{}_{}_{}.mp4'.format(yid,start_time,end_time))
    if os.path.exists(out_fn):
        return
    # Get video and extract frames in 30 fps
    #ffmpeg -i "${video}" -r 30 -q:v 1 "${out_name}"
    command = ['ffmpeg',
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-i', '"%s"' % tmp_fn,
               '-c:v', 'copy', '-c:a', 'copy', # libx264 
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % out_fn]
    command = ' '.join(command)
    try:
        #output = subprocess.check_output(command, shell=True,
        #                                 stderr=subprocess.STDOUT)
        output = subprocess.run(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print(yid, err.output)
        return #False, err.output
    return #True, 'Done'

def process_files(input_csv, input_xlsx):
    '''
    args
        input_csv: str, input csv filename
        input_xlsx: str, input xlsx filename
    return
        dataset: [{youtube_id, title, start_time, end_time}]
    '''
    xlsx = pd.read_excel(input_xlsx)
    datas = xlsx.to_numpy()
    yid2title = {}
    for elem in datas:
        yid2title[elem[0]] = elem[3]
        
    data_csv = pd.read_csv(input_csv)
    data_npy = data_csv.to_numpy()
    
    dataset = []
    for elem in data_npy:
        if elem[2] not in yid2title:
            continue
        dataset.append({
            'yid': elem[2],
            'title': yid2title[elem[2]],
            'start_time': elem[3],
            'end_time': elem[4]
        })
    
    return dataset

def create_video_folders(out_dir):
    """Creates a directory for each label name in the dataset."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


def main(input_csv, input_xlsx, tmp_dir, out_dir, num_worker):
    #start_time = time.time()
    create_video_folders(out_dir)
    
    dataset = process_files(input_csv, input_xlsx)
    dataset = list(enumerate(dataset))
    print("total", len(dataset))
    #with Pool(num_worker) as p:
    #    p.map(partial(split_vid, tmp_dir=tmp_dir, out_dir=out_dir), dataset)
    for elem in tqdm(dataset):
        split_vid(elem, tmp_dir, out_dir)
    #print("\n end : {}s \n".format(time.time() - start_time))
        
        
if __name__ == '__main__':
    description = 'Helper script for extract frames.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--input_csv', type=str, default='./data_demo/KDX_demo.csv',
                   help=('CSV file containing the following format: '
                         'label,youtube_id,time_start,time_end'))
    p.add_argument('--input_xlsx', type=str, default='./data_demo/KDX_YoutubeData.xlsx',
                   help=('xlsx file containing the program info'))
    p.add_argument('--tmp_dir', type=str, default='./data_demo/raw_videos',
                   help=('Output directory where raw videos are saved'))
    p.add_argument('--out_dir', type=str, default='./data_demo/videos',
                   help='Output directory where splitted video will be saved.')
    p.add_argument('--num_worker', type=int, default=4,
                   help='Number of workers')
    main(**vars(p.parse_args()))
