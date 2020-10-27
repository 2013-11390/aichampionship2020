import pickle
import torch
import json
import pandas as pd
import os
import argparse
import slowfast.utils.metrics as metrics


parser = argparse.ArgumentParser(
    description="Make csv file."
)
parser.add_argument(
    '--test_dirs',
    type=str, nargs='+',
    default=['slowfast_r101/test_3x10',
            # 'slowfast_r101_focal/test_3x10',
            # 'slowfast_r101_sample160/test_3x10'
            ],
    help='Test dirs to make csv'
)

args = parser.parse_args()

test_dirs = args.test_dirs

with open('./data/test_anno.json', 'r') as f:
    test_anno_json = json.load(f)

label_class, vids, labels = [], [], []
for lidx, elem in enumerate(test_anno_json):
    label_class.append(elem['label'])
    labels += [lidx] * len(elem['vids'])
    vids += elem['vids']

pred_logits = torch.zeros((len(vids), len(label_class))) # [V x C]
for test_dir in test_dirs:
    with open(os.path.join('logdir', test_dir, 'pred_logits.pkl'), 'rb') as f:
        pred_dict = pickle.load(f)
    for vidx, vid in enumerate(vids):
        pred_logits[vidx] += pred_dict[vid].squeeze()

labels_torch = torch.LongTensor(labels)
top1_tensor, top2_tensor = metrics.topks_correct(pred_logits, labels_torch, (1, 2))
print_str = "top1: %.5f top2: %.5f\n"%(top1_tensor.mean(), top2_tensor.mean())
anno5_right_1, anno5_right_2, anno5_num = 0, 0, 0
for lidx, name in enumerate(label_class):
    mask = (labels_torch == lidx).float()
    top1 = (top1_tensor * mask).sum() / mask.sum()
    top2 = (top2_tensor * mask).sum() / mask.sum()
    print_str += name + " top1: %.5f top2: %.5f num: %d\n"%(top1, top2, mask.sum())
    if label_class[lidx] in  ['서있다','걷다','뛰다','앉아있다','누워있다']:
        anno5_right_1 += (top1_tensor * mask).sum()
        anno5_right_2 += (top2_tensor * mask).sum()
        anno5_num += mask.sum()
print("anno5 top1: %.5f top2: %.5f"%(anno5_right_1/anno5_num, anno5_right_2/anno5_num))
print(print_str)

# Make test csv
pred_label_names = []
pred_indices = pred_logits.argmax(1) # V
for pred_idx in pred_indices:
    pred_label_names.append(label_class[pred_idx])
    
test_df = {'label':  pred_label_names,
           'video_id': vids
          }

test_df = pd.DataFrame (test_df, columns = ['label', 'video_id'])
test_df.to_csv('test_pred.csv', index=False)
