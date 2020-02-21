#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging
import os
import sys
import warnings

# disable warnings when loading tensorboard
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_value_
# below import will lead to PermissionError on Xiaomi cluster sometimes, need to check.
# from torch.utils.tensorboard import SummaryWriter 

import kaldi
import kaldi_pybind.chain as chain
import kaldi_pybind.fst as fst

from chain_loss import KaldiChainObjfFunction
from common import load_checkpoint
from common import save_checkpoint
from common import save_training_info
from common import setup_logger
from egs_dataset import get_egs_dataloader
from model import get_chain_model
from options import get_args

def auto_alloc_device(num: int = 1):
    """
    allocate devices according to if we can put a tensor on a device.
    assuming the cuda cards are in 'Exclusive_Process' compute mode.
    :param num: numbers of gpu cards to allocate
    :return: a list of tuple(device, place_holder_var) allocated
    """

    dev_list = []
    if not torch.cuda.is_available():
        return dev_list
    for i in range(torch.cuda.device_count()):
        if len(dev_list) >= num:
            break
        dev = torch.device("cuda:" + str(i))
        x = torch.zeros(1)
        try:
            x = x.to(dev)
            dev_list.append((i, x))
        except RuntimeError:  # RuntimeError: CUDA error: all CUDA-capable devices are busy or unavailable
            pass
    return dev_list


def get_validation_objf(dataloader, model, device, criterion, opts, den_graph):
    total_objf = 0.
    total_weight = 0.
    total_frames = 0.  # for display only

    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        key_list, feature_list, supervision_list = batch

        assert len(key_list) == len(feature_list) == len(supervision_list)
        batch_size = len(key_list)

        for n in range(batch_size):
            feats = feature_list[n]
            assert feats.ndim == 3

            # at this point, feats is [N, T, C]
            feats = feats.to(device)

            with torch.no_grad():
                nnet_output, xent_output = model(feats)

            # at this point, nnet_output is: [N, T, C]
            # refer to kaldi/src/chain/chain-training.h
            # the output should be organized as
            # [all sequences for frame 0]
            # [all sequences for frame 1]
            # [etc.]
            nnet_output = nnet_output.permute(1, 0, 2)
            # at this point, nnet_output is: [T, N, C]
            nnet_output = nnet_output.contiguous().view(-1,
                                                        nnet_output.shape[-1])

            # at this point, xent_output is: [N, T, C]
            xent_output = xent_output.permute(1, 0, 2)
            # at this point, xent_output is: [T, N, C]
            xent_output = xent_output.contiguous().view(-1,
                                                        xent_output.shape[-1])
            objf_l2_term_weight = criterion(opts, den_graph,
                                            supervision_list[n], nnet_output,
                                            xent_output)
            objf = objf_l2_term_weight[0]

            objf_l2_term_weight = objf_l2_term_weight.cpu()

            total_objf += objf_l2_term_weight[0].item()
            total_weight += objf_l2_term_weight[2].item()

            num_frames = nnet_output.shape[0]
            total_frames += num_frames

    return total_objf, total_weight, total_frames


def train_one_epoch(dataloader, valid_dataloader, model, device, optimizer,
                    criterion, current_epoch, opts, den_graph, tf_writer, rank):
    model.train()

    total_objf = 0.
    total_weight = 0.
    total_frames = 0.  # for display only

    for batch_idx, batch in enumerate(dataloader):
        key_list, feature_list, supervision_list = batch
        assert len(key_list) == len(feature_list) == len(supervision_list)
        batch_size = len(key_list)
        for n in range(batch_size):
            feats = feature_list[n]
            assert feats.ndim == 3

            # at this point, feats is [N, T, C]
            feats = feats.to(device)
            nnet_output, xent_output = model(feats)

            # at this point, nnet_output is: [N, T, C]
            # refer to kaldi/src/chain/chain-training.h
            # the output should be organized as
            # [all sequences for frame 0]
            # [all sequences for frame 1]
            # [etc.]
            nnet_output = nnet_output.permute(1, 0, 2)
            # at this point, nnet_output is: [T, N, C]
            nnet_output = nnet_output.contiguous().view(-1,
                                                        nnet_output.shape[-1])

            # at this point, xent_output is: [N, T, C]
            xent_output = xent_output.permute(1, 0, 2)
            # at this point, xent_output is: [T, N, C]
            xent_output = xent_output.contiguous().view(-1,
                                                        xent_output.shape[-1])
            objf_l2_term_weight = criterion(opts, den_graph,
                                            supervision_list[n], nnet_output,
                                            xent_output)
            objf = objf_l2_term_weight[0]
            optimizer.zero_grad()
            objf.backward()

            clip_grad_value_(model.parameters(), 5.0)

            optimizer.step()

            objf_l2_term_weight = objf_l2_term_weight.detach().cpu()

            total_objf += objf_l2_term_weight[0].item()
            total_weight += objf_l2_term_weight[2].item()
            num_frames = nnet_output.shape[0]
            total_frames += num_frames

        if batch_idx % 100 == 0:
            logging.info(
                'Device ({}) processing {}/{}({:.6f}%) global average objf: {:.6f} over {} '
                'frames, current batch average objf: {:.6f} over {} frames, epoch {}'
                .format(
                    device.index, batch_idx, len(dataloader),
                    float(batch_idx) / len(dataloader) * 100,
                    total_objf / total_weight, total_frames,
                    objf_l2_term_weight[0].item() /
                    objf_l2_term_weight[2].item(), num_frames, current_epoch))

        if valid_dataloader and batch_idx % 1000 == 0:
            total_valid_objf, total_valid_weight, total_valid_frames = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                device=device,
                criterion=criterion,
                opts=opts,
                den_graph=den_graph)

            model.train()

            logging.info(
                'Validation average objf: {:.6f} over {} frames'.format(
                    total_valid_objf / total_valid_weight, total_valid_frames))
            if tf_writer:
                tf_writer.add_scalar('train/global_valid_average_objf',
                                 total_valid_objf / total_valid_weight,
                                 batch_idx + current_epoch * len(dataloader))

        if rank == 0 and batch_idx % 100 == 0 and tf_writer != None:
            tf_writer.add_scalar('train/global_average_objf',
                                 total_objf / total_weight,
                                 batch_idx + current_epoch * len(dataloader))
            tf_writer.add_scalar(
                'train/current_batch_average_objf',
                objf_l2_term_weight[0].item() / objf_l2_term_weight[2].item(),
                batch_idx + current_epoch * len(dataloader))

            state_dict = model.state_dict()
            for key, value in state_dict.items():
                # skip batchnorm parameters
                if value.dtype != torch.float32:
                    continue
                if 'running_mean' in key or 'running_var' in key:
                    continue

                with torch.no_grad():
                    frobenius_norm = torch.norm(value, p='fro')

                tf_writer.add_scalar(
                    'train/parameters/{}'.format(key), frobenius_norm,
                    batch_idx + current_epoch * len(dataloader))

    return total_objf / total_weight


def main():
    os.environ["NCCL_IB_DISABLE"]="1"  
    
    args = get_args()
    if torch.cuda.is_available() == False:
        sys.exit(-1)
    
    devs = auto_alloc_device()
    if len(devs) <= 0:
        print('allocate gpu failed')
        sys.exit(-1)
    device_id = devs[0][0] # args.device_id

    local_rank = int(os.environ['SGE_TASK_ID']) - 1
    setup_logger('{}/log-train-rank-{}'.format(args.dir, local_rank),
                 args.log_level)
    logging.info(' '.join(sys.argv))

    # tcp://127.0.0.1:7275
    # init_method='file:///home/storage23/qiuhaowen/kaldi/egs/aishell/s10/init.log',
    dist.init_process_group('nccl',
                            init_method=args.init_method,
                            rank=local_rank,
                            world_size=args.world_size)

    # WARNING(fangjun): we have to select GPU at the very
    # beginning; otherwise you will get trouble later
    kaldi.SelectGpuDevice(device_id=device_id)
    kaldi.CuDeviceAllowMultithreading()

    device = torch.device('cuda', device_id)

    den_fst = fst.StdVectorFst.Read(args.den_fst_filename)

    opts = chain.ChainTrainingOptions()
    opts.l2_regularize = args.l2_regularize
    opts.xent_regularize = args.xent_regularize
    opts.leaky_hmm_coefficient = args.leaky_hmm_coefficient

    den_graph = chain.DenominatorGraph(fst=den_fst, num_pdfs=args.output_dim)

    model = get_chain_model(
        feat_dim=args.feat_dim,
        output_dim=args.output_dim,
        lda_mat_filename=args.lda_mat_filename,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        prefinal_bottleneck_dim=args.prefinal_bottleneck_dim,
        kernel_size_list=args.kernel_size_list,
        subsampling_factor_list=args.subsampling_factor_list)

    start_epoch = 0
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    best_objf = -100000

    if args.checkpoint:
        start_epoch, learning_rate, best_objf = load_checkpoint(
            args.checkpoint, model)
        logging.info(
            'Device ({device_id}) loaded from checkpoint: start epoch {start_epoch}, '
            'learning rate {learning_rate}, best objf {best_objf}'.format(
                device_id=device_id,
                start_epoch=start_epoch,
                learning_rate=learning_rate,
                best_objf=best_objf))

    model.to(device)

    model = DDP(model, device_ids=[device_id])

    dataloader = get_egs_dataloader(egs_dir_or_scp=args.cegs_dir,
                                    egs_left_context=args.egs_left_context,
                                    egs_right_context=args.egs_right_context,
                                    shuffle=False,
                                    world_size=args.world_size,
                                    local_rank=local_rank)

    if local_rank == 0:
        valid_dataloader = get_egs_dataloader(
            egs_dir_or_scp=args.valid_cegs_scp,
            egs_left_context=args.egs_left_context,
            egs_right_context=args.egs_right_context,
            shuffle=False)
    else:
        valid_dataloader = None

    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=5e-4)

    criterion = KaldiChainObjfFunction.apply

    if local_rank == 0:
        # tf_writer = SummaryWriter(log_dir='{}/tensorboard'.format(args.dir))
        tf_writer = None
    else:
        tf_writer = None

    best_epoch = start_epoch
    best_model_path = os.path.join(args.dir, 'best_model.pt')
    best_epoch_info_filename = os.path.join(args.dir, 'best-epoch-info')

    dist.barrier()

    try:
        for epoch in range(start_epoch, args.num_epochs):
            learning_rate =  args.learning_rate * pow(0.4, epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            logging.info('epoch {}, learning rate {}'.format(
                epoch, learning_rate))

            if tf_writer:
                tf_writer.add_scalar('learning_rate', learning_rate, epoch)

            objf = train_one_epoch(dataloader=dataloader,
                                   valid_dataloader=valid_dataloader,
                                   model=model,
                                   device=device,
                                   optimizer=optimizer,
                                   criterion=criterion,
                                   current_epoch=epoch,
                                   opts=opts,
                                   den_graph=den_graph,
                                   tf_writer=tf_writer,
                                   rank=local_rank)

            if best_objf is None:
                best_objf = objf
                best_epoch = epoch

            # the higher, the better
            if objf > best_objf:
                best_objf = objf
                best_epoch = epoch
                save_checkpoint(filename=best_model_path,
                                model=model,
                                epoch=epoch,
                                learning_rate=learning_rate,
                                objf=objf,
                                local_rank=local_rank)
                save_training_info(filename=best_epoch_info_filename,
                                   model_path=best_model_path,
                                   current_epoch=epoch,
                                   learning_rate=learning_rate,
                                   objf=best_objf,
                                   best_objf=best_objf,
                                   best_epoch=best_epoch,
                                   local_rank=local_rank)

            # we always save the model for every epoch
            model_path = os.path.join(args.dir, 'epoch-{}.pt'.format(epoch))
            save_checkpoint(filename=model_path,
                            model=model,
                            epoch=epoch,
                            learning_rate=learning_rate,
                            objf=objf,
                            local_rank=local_rank)

            epoch_info_filename = os.path.join(args.dir,
                                               'epoch-{}-info'.format(epoch))
            save_training_info(filename=epoch_info_filename,
                               model_path=model_path,
                               current_epoch=epoch,
                               learning_rate=learning_rate,
                               objf=objf,
                               best_objf=best_objf,
                               best_epoch=best_epoch,
                               local_rank=local_rank)

    except KeyboardInterrupt:
        # save the model when ctrl-c is pressed
        model_path = os.path.join(args.dir,
                                  'epoch-{}-interrupted.pt'.format(epoch))
        # use a very small objf for interrupted model
        objf = -100000
        save_checkpoint(model_path,
                        model=model,
                        epoch=epoch,
                        learning_rate=learning_rate,
                        objf=objf,
                        local_rank=local_rank)

        epoch_info_filename = os.path.join(
            args.dir, 'epoch-{}-interrupted-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           best_epoch=best_epoch,
                           local_rank=local_rank)

    if tf_writer:
        tf_writer.close()
    logging.warning('Done')


if __name__ == '__main__':
    torch.manual_seed(20191227)
    main()
