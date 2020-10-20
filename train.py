import argparse
import torch
import pickle
import numpy as np
import os
import random
from shutil import copyfile
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

def train_epoch(train_dloader, model, optimizer, cur_epoch, cfg):
    model.train()
    train_tqdm = tqdm(train_dloader, ncols=80)
    data_size = len(train_dloader)
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_tqdm):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
                    
        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)
        
        preds = model(inputs)
            
        # Explicitly declare reduction to mean.
        loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

        # Compute the loss.
        loss = loss_fun(preds, labels)

        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()
        
        train_tqdm.set_description("Train_loss: %.4f" % loss.cpu().item())
        
def eval_epoch(val_dloader, model, cur_epoch, cfg):
    model.eval()
    results = defaultdict(list)
    for cur_iter, (inputs, labels, _, meta) in enumerate(tqdm(val_dloader, ncols=80)):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
        with torch.no_grad():
            preds = model(inputs)
        top1_tensor, top5_tensor = metrics.topks_correct(preds, labels, (1, 5))
        if cfg.NUM_GPUS:
            top1_tensor, top5_tensor = top1_tensor.cpu(), top5_tensor.cpu()
        results['top1'] += top1_tensor.tolist()
        results['top5'] += top5_tensor.tolist()
    print_str = "epoch: {} ".format(cur_epoch)
    for key in results:
        results[key] = sum(results[key]) / len(results[key])
        print_str += "{}: {} ".format(key, results[key])
    print(print_str)
    with open(os.path.join(cfg.OUTPUT_DIR, 'res_out.txt'), 'a') as f:
        f.write("epoch: " + str(cur_epoch) + " " + print_str + "\n")
    return results
            
def train(cfg):
    # Build model
    model = build_model(cfg)
    optimizer = optim.construct_optimizer(model, cfg)
    # load checkpoint
    start_epoch = cu.load_train_checkpoint(cfg, model, optimizer)

    # Build data loader
    train_loader = dataloader.construct_loader(cfg, "train")
    val_loader = dataloader.construct_loader(cfg, "val")
    precise_bn_loader = dataloader.construct_loader(cfg, "train")
    
    best_accuracy = 0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Train for one epoch.
        train_epoch(train_loader, model, optimizer, cur_epoch, cfg)
        
        is_eval_epoch = cur_epoch > 0
        # Compute precise BN stats.
        if (is_eval_epoch
            and cfg.BN.USE_PRECISE_STATS
            and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = aggregate_sub_bn_stats(model) # for SubBatchNorm3d call before eval

        # Evaluate the model on validation set.
        if is_eval_epoch:
            results = eval_epoch(val_loader, model, cur_epoch, cfg)
            accuracy = results['top1']
            if accuracy > best_accuracy:
                print("*** Saving best ****")
                best_accuracy = accuracy
                torch.save({'epoch': cur_epoch + 1,
                            'model_state': model.state_dict(),
                            'optimizer_state' : optimizer.state_dict()},
                             os.path.join(cfg.OUTPUT_DIR, 'best_ckpt.pth'))


def aggregate_sub_bn_stats(module):
    """
    Recursively find all SubBN modules and aggregate sub-BN stats.
    Args:
        module (nn.Module)
    Returns:
        count (int): number of SubBN module found.
    """
    count = 0
    for child in module.children():
        if isinstance(child, SubBatchNorm3d):
            child.aggregate_stats()
            count += 1
        else:
            count += aggregate_sub_bn_stats(child)
    return count

def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Provide video training and testing pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/SLOWFAST_8x8_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "--tag",
        help="tag",
        default="slowfast_8x8_r50",
        type=str,
    )

    args = parser.parse_args()
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    
    cfg.OUTPUT_DIR = os.path.join('logdir', args.tag)
    # Make dir
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    # Save cfg
    copyfile(args.cfg_file, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))
    #with open(os.path.join(cfg.OUTPUT_DIR, 'config.pkl'), 'wb') as f:
    #    pickle.dump(cfg, f)
    
    # Setup seed
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    
    train(cfg)