#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import random

import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.data.cvit.pretrain_utils import *
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def vocab_code():
    
    print ("Executing the vocab checkpoint code")
    
    #Task 2
    french_english_file = open("/content/drive/My Drive/RA-Project-IIIT-H/Cat_Forgetting/ilmulti-master/resources/cat_forgetting_dicts/en-fr.dict.txt", 'r')
    dict_fr_text = french_english_file.readlines()
    
    #Task 1
    spanish_english_file = open("/content/drive/My Drive/RA-Project-IIIT-H/Cat_Forgetting/ilmulti-master/resources/cat_forgetting_dicts/en-es.dict.txt", 'r')
    dict_es_text = spanish_english_file.readlines()
    
    #Combined tasks dictionary
    combined_dictionary = open("/content/drive/My Drive/RA-Project-IIIT-H/Cat_Forgetting/ilmulti-master/resources/cat_forgetting_dicts/en-fr-es.dict.txt", 'r')
    dict_es_fr_text = combined_dictionary.readlines()
    
    dict_fr = dict()
    dict_es = dict()
    dict_comb = dict()
    
    for i in range(len(dict_fr_text)):
        a, b = dict_fr_text[i].strip().split()
        dict_fr[a] = int(b)
    
    for i in range(len(dict_es_text)):
        a, b = dict_es_text[i].strip().split()
        dict_es[a] = int(b)
        
    for i in range(len(dict_es_fr_text)):
        a, b = dict_es_fr_text[i].strip().split()
        dict_comb[a] = int(b)
        
    intersecting_keys = set(dict_fr.keys()) & set(dict_es.keys())
    
    indices_dict_combined = dict()
    indices_dict_original = dict()

    #Intersect 
    for i in range(len(list(intersecting_keys))):
        if list(intersecting_keys)[i] in dict_comb:
            indices_dict_combined[list(intersecting_keys)[i]] = list(dict_comb.keys()).index(list(intersecting_keys)[i])

        if list(intersecting_keys)[i] in dict_es:
            indices_dict_original[list(intersecting_keys)[i]] = list(dict_es.keys()).index(list(intersecting_keys)[i])
    
    # print ("Inside the vocab code function", indices_dict_original)
    # print ("Inside the vocab code function", indices_dict_combined)
    return (indices_dict_original, indices_dict_combined)
    
    
def model_initializing(args, trainer, trainer_proxy, indices_dict_original, indices_dict_combined):
    
    print ("Initializing the model with the old checkpoint")
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint_vocab_loading(args, trainer_proxy)
    # for param_tensor in trainer_proxy.get_model().state_dict():
    #     print(param_tensor, "\t", trainer_proxy.get_model().state_dict()[param_tensor].size())
    
    # print ("Exiting")
    # import sys
    # sys.exit(0)
    
    # decoder.embed_tokens.weight 	 torch.Size([14880, 512])
    # decoder.embed_positions._float_tensor 	 torch.Size([1])
    
    encoding_proxy_vector = trainer_proxy.get_model().state_dict()['encoder.embed_tokens.weight']
    decoding_proxy_vector = trainer_proxy.get_model().state_dict()['decoder.embed_tokens.weight']
    
    encoding_current_vector = trainer.get_model().state_dict()['encoder.embed_tokens.weight']
    decoding_current_vector = trainer.get_model().state_dict()['decoder.embed_tokens.weight']
    
    print ("Fitting the old values in the new ones now")
    
    for key, value in indices_dict_combined.items():
        original_value = indices_dict_original[key]
        new_value = value 
        
        encoding_current_vector[new_value, :].copy_(encoding_proxy_vector[original_value, :])
        decoding_current_vector[new_value, :].copy_(decoding_proxy_vector[original_value, :])

    trainer.get_model().state_dict()['encoder.embed_tokens.weight'].copy_(encoding_current_vector)
    trainer.get_model().state_dict()['decoder.embed_tokens.weight'].copy_(decoding_current_vector)
    
    trainer.get_model().state_dict()['encoder.embed_positions._float_tensor'].copy_(trainer_proxy.get_model().state_dict()['encoder.embed_positions._float_tensor'])
    trainer.get_model().state_dict()['decoder.embed_positions._float_tensor'].copy_(trainer_proxy.get_model().state_dict()['decoder.embed_positions._float_tensor'])
    
    #print (trainer_proxy.get_model().state_dict().keys())
    #print (trainer.get_model().state_dict()['model'])
    
    trainer._build_optimizer()
    epoch_itr = trainer.get_train_iterator(epoch=0)
    val_loss = 1.000
    
    extra_state = {
        'train_iterator': epoch_itr.state_dict(),
        'val_loss': val_loss,
    }
    
    trainer.save_checkpoint(os.path.join(args.save_dir, 'checkpoint_last.pt'), extra_state)
    # state_dict = dict()
    # state_dict['model'] = trainer.get_model().state_dict()
    # state_dict['args'] = args
    # state_dict['best_loss'] = 1.000
    
    # import os
    # checkpoint_utils.torch_persistent_save(state_dict, os.path.join(args.save_dir, 'checkpoint_last.pt'))
    # # trainer.save_checkpoint(os.path.join(args.save_dir, 'checkpoint_last.pt'), None)
    
    # print ("Inside the model initializing function")
    # print (trainer.get_model().state_dict()['encoder.embed_tokens.weight'])
    # print (trainer_proxy.get_model().state_dict()['encoder.embed_tokens.weight'])
    # print (trainer.get_model().state_dict()['encoder.embed_positions._float_tensor'])
    # print (trainer_proxy.get_model().state_dict()['encoder.embed_positions._float_tensor'])
    
    # print ("\n")
    # print (trainer.get_model().state_dict()['decoder.embed_tokens.weight'])
    # print (trainer_proxy.get_model().state_dict()['decoder.embed_tokens.weight'])
    # print (trainer.get_model().state_dict()['decoder.embed_positions._float_tensor'])
    # print (trainer_proxy.get_model().state_dict()['decoder.embed_positions._float_tensor'])
    
    

def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    
    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
        
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)
    
    #print ("Hitting the if clause")
    #print (args.flag_vocab_loading)
    if (args.flag_vocab_loading == 2):
        
        #vocab code
        indices_dict_original, indices_dict_combined = vocab_code()
        # print ("Inside the main function", indices_dict_original)
        # print ("Inside the main function", indices_dict_combined)
        
        #print ("Is this executing")
        task = tasks.setup_task(args, flag = 1)
        task_proxy = tasks.setup_task(args, flag = 2)
        
        model = task.build_model(args)
        model_proxy = task_proxy.build_model(args)
        
        criterion = task.build_criterion(args)
        criterion_proxy = task_proxy.build_criterion(args)
        
        # print("The new model is", model)
        # print ("\n")
        # print ("The old model is", model_proxy)
        
        trainer = Trainer (args, task, model, criterion)
        trainer_proxy = Trainer (args, task_proxy, model_proxy, criterion_proxy)
        
        print ("Inside the main function")
        print (trainer.get_model().state_dict()['encoder.embed_tokens.weight'])
        print ("/n")
    
        model_initializing(args, trainer, trainer_proxy, indices_dict_original, indices_dict_combined)
        
        print ("Exiting")
        import sys
        sys.exit(0)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args, flag = 1)
    

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)

    criterion = task.build_criterion(args)

    if (args.task == 'pretrain_lang_modeling'):
        split_create(model)
    
    print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))
    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)
    

    # Train until the learning rate gets too small
    args.max_epoch = 10
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        # TODO(jerin): 
        # ~The fuck. This works.~ Doesn't work.
        # https://github.com/pytorch/fairseq/issues/232
        # torch.cuda.empty_cache()
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    #print ("Inside cli main", parser)
    args = options.parse_args_and_arch(parser)
    #print ("Inside cli main", args)
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
