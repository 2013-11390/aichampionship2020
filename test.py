import argparse
import torch
import pickle
import numpy as np
import os
from shutil import copyfile
import random
from tqdm import tqdm
from collections import defaultdict
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats

from slowfast.config.defaults import get_cfg
import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
from slowfast.models import build_model
from slowfast.models.batchnorm_helper import SubBatchNorm3d
import slowfast.utils.checkpoint as cu
import slowfast.utils.metrics as metrics
import slowfast.datasets.dataloader as dataloader

def perform_test(test_dloader, model, cfg):
    model.eval()
    ens_number = cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
    # Collect predictions
    collect_pred = []
    collect_vids = []
    for cur_iter, (inputs, labels, vids, extra_data) in enumerate(tqdm(test_dloader, ncols=80)):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=cfg.DATA_LOADER.PIN_MEMORY)
            else:
                inputs = inputs.cuda(non_blocking=cfg.DATA_LOADER.PIN_MEMORY)
            for key, val in extra_data.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    extra_data[key] = val.cuda(non_blocking=True)
        with torch.no_grad():
            if cfg.DETECTION.ENABLE:
                preds = model(inputs, extra_data["boxes"])
            else:
                preds = model(inputs)  
            preds = preds.cpu()
        collect_pred.append(preds)
        collect_vids.append(vids)
    # Make vid2label
    idx2label, label2idx = test_dloader.dataset.idx2label, test_dloader.dataset.label2idx
    test_datas = test_dloader.dataset.data
    vid2label = {}
    for label, vid, _ in test_datas:
        if vid not in vid2label:
            vid2label[vid] = torch.LongTensor([label2idx[label]])
    # Gather vid2logits
    vid2logits = {}
    for preds, vids in tqdm(zip(collect_pred, collect_vids)):
        B = preds.size(0)
        for b in range(B):
            if vids[b] not in vid2logits:
                vid2logits[vids[b]] = preds[b]
            else:
                if cfg.TEST.ENSEMBLE_METHOD == 'sum':
                    vid2logits[vids[b]] += preds[b]
                elif cfg.TEST.ENSEMBLE_METHOD == 'max':
                    vid2logits[vids[b]], _ = torch.stack((vid2logits[vids[b]], preds[b])).max(0)
    # Calculate accuracy
    results = defaultdict(list)
    for vid in vid2label:
        #vid2logits[vid] = torch.tensor(vid2logits[vid])
        if cfg.TEST.ENSEMBLE_METHOD == 'sum':
            vid2logits[vid] = vid2logits[vid] / ens_number
        preds = vid2logits[vid].unsqueeze(0)
        labels = vid2label[vid].unsqueeze(0)
        top1_tensor, top2_tensor = metrics.topks_correct(preds, labels, (1, 2))
        results['top1'] += top1_tensor.tolist()
        results['top2'] += top2_tensor.tolist()
        #results['top1_'+idx2label[labels[0]]] += top1_tensor.tolist()
        #results['top2_'+idx2label[labels[0]]] += top2_tensor.tolist()
    print_str = "test: "
    for key in results:
        results[key] = sum(results[key]) / len(results[key])
        print_str += "{}: {} \n".format(key, results[key])
    print(print_str)
    with open(os.path.join(cfg.TEST_OUTPUT_DIR, 'res_out.txt'), 'a') as f:
        f.write(print_str)
    with open(os.path.join(cfg.TEST_OUTPUT_DIR, 'pred_logits.pkl'), 'wb') as f:
        pickle.dump(vid2logits, f)
    return results


def test(cfg):
    # Build model
    model = build_model(cfg)
    optimizer = optim.construct_optimizer(model, cfg)
    # load checkpoint
    start_epoch = cu.load_test_checkpoint(cfg, model)
    print("Load model epoch", start_epoch)
    
    # Build data loader
    test_loader = dataloader.construct_loader(cfg, "test")
    
    # Perform test
    results = perform_test(test_loader, model, cfg)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Provide video training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/test_config.yaml",
        type=str,
    )
    parser.add_argument(
        "--tag",
        help="tag",
        default="slowfast_r101",
        type=str,
    )
    parser.add_argument(
        "--test_tag",
        help="test_tag",
        default="1",
        type=str,
    )

    args = parser.parse_args()
    cfg = get_cfg()
    # Merge train configs
    cfg.merge_from_file(os.path.join('logdir', args.tag, 'config.yaml'))
    # Merge test configs
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    
    print("Test using", args.tag)
    cfg.OUTPUT_DIR = os.path.join('logdir', args.tag)
    cfg.TEST_OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "test_"+args.test_tag)
    # Make dir
    if not os.path.exists(cfg.TEST_OUTPUT_DIR):
        os.makedirs(cfg.TEST_OUTPUT_DIR)
    # Save test cfg
    copyfile(args.cfg_file, os.path.join(cfg.TEST_OUTPUT_DIR, 'test_config.yaml'))
    
    # Setup seed
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    
    test(cfg)
