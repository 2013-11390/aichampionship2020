# Import dependency
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import random
import os
import argparse
from IPython.display import Video
random.seed(2020)

def main(output_dir):
    # Load dataframe
    df = pd.read_csv('./data/KDX_train.csv')

    # Remove duplication
    data = {}
    df_npy = df.to_numpy()
    print("raw_data length", len(df_npy))
    for elem in df_npy:
        vid = "{}_{}_{}".format(elem[2],elem[3],elem[4])
        # Check just duplication or multi-label
        if vid in data and elem[1].strip() != data[vid]:
            print(vid, elem[1].strip(), data[vid])
        label =  elem[1].strip()
        if label == '입술을 바르다':
            label = '입술을 바른다'
        if label == '불을피우다':
            label = '불을 피우다'
        if label == '야채를 칼질히다':
            label = '야채를 칼질하다'
        if label == '얼굴에 바르다':
            label = '얼굴에 바른다'
        data[vid] = label
    print("After removing duplications", len(data))
    
    # Check available
    data_ = {}
    for vid in data:
        if os.path.exists('./data/raw_frames/'+vid):
            data_[vid] = data[vid]
    data = data_
    print("After collecting availables", len(data))

    # Select labels
    label_data = defaultdict(list)
    for vid in data:
        label_data[data[vid]].append(vid)

    selected_labels = [key for key in label_data if len(label_data[key])>=50]
    print("selected class num", len(selected_labels))

    # Build annotation
    train_data, val_data, test_data = [], [], []
    for label in selected_labels:
        datas = label_data[label]
        test_num = len(datas) // 5
        val_num = len(datas) // 10
        random.shuffle(datas)
        test_data.append({'label': label,
                          'vids': datas[:test_num]})
        val_data.append({'label': label,
                         'vids': datas[test_num:test_num+val_num]})
        train_data.append({'label': label,
                           'vids': datas[test_num+val_num:]})

    # Save annotations
    with open(os.path.join(output_dir, 'train_anno.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(output_dir, 'val_anno.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(os.path.join(output_dir, 'test_anno.json'), 'w') as f:
        json.dump(test_data, f, indent=2)

    # Make test csv
    label_list = []
    vid_list = []
    for elem in test_data:
        for vid in elem['vids']:
            label_list.append(elem['label'])
            vid_list.append(vid)
    test_df = {'label':  label_list,
               'video_id': vid_list
              }

    test_df = pd.DataFrame (test_df, columns = ['label', 'video_id'])
    test_df.to_csv(os.path.join(output_dir, 'test_gt.csv'), index=False)

if __name__ == '__main__':
    description = 'Helper script for make annotation.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--output_dir', type=str, default='./data',
                   help='Output directory where videos will be saved.')

    main(**vars(p.parse_args()))