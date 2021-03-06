"""
Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> [--test-src=<file> --test-tgt=<file>] --vocab=<file> --word_freq=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --word_freq=<file>                      word freq file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 5]
    --warmup-iters=<int>                    warmup iterations [default: 5000]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --bleu-niter=<int>                      perform BLEU evaluation on test set every N inters [default: 500]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --order-name=<str>                      ordering function name [default: none]
    --pacing-name=<str>                     pacing function name [default: none]
    --ignore-test-bleu=<int>                whether or not to ignore bleu scores (train_local)
"""
import math
import sys
import pickle
import time


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from model import Hypothesis, NMT, TransformerNMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils
from torch.utils.tensorboard import SummaryWriter

from scoring import load_order, balance_order, visualize_scoring
from pacing import pacing_data
from utils import get_pacing_batch

import gc

def write_to_file(train_data):
    source_sentences = [x[0] for x in train_data]
    target_sentences = [x[1] for x in train_data]

    len_source = len(source_sentences)
    len_target = len(target_sentences)

    assert(len_source == len_target), print ("Something's gone wrong mate, length's not the same")

    exemplar_length = int(0.10*len_source)

    exemplar_source_sentences = source_sentences[:exemplar_length]
    exemplar_target_sentences = target_sentences[:exemplar_length]

    with open("exemplar_sources.en-fr.fr", mode = 'w+', encoding = 'utf-8') as f:

        for list_sentence in exemplar_source_sentences:
            sentence = " ".join(list_sentence)
            f.write(sentence+"\n")

        f.close()

    with open("exemplar_targets.en-fr.en", mode = 'w+', encoding = 'utf-8') as f:

        for list_sentence in exemplar_target_sentences:
            
            #Get rid of the tags
            list_sentence.pop(0)
            list_sentence.pop(-1)
            
            sentence = " ".join(list_sentence)
            f.write(sentence+"\n")

        f.close()
        

def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(
                len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def tuple_length_ok(src_tok: list, tgt_tok: list, limit: int):
    return len(src_tok) < limit and len(tgt_tok) < limit


def clean_data(data: list, limit: int):
    return list(filter(lambda t: tuple_length_ok(t[0], t[1], limit), data))


dev_mode = False

def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    do_bleu = '--ignore-test-bleu' not in args or not args['--ignore-test-bleu']
    train_data_src = read_corpus(
        args['--train-src'], source='src', dev_mode=dev_mode, space_tokenize=True)
    train_data_tgt = read_corpus(
        args['--train-tgt'], source='tgt', dev_mode=dev_mode, space_tokenize=True)

    dev_data_src = read_corpus(
        args['--dev-src'], source='src', dev_mode=dev_mode, space_tokenize=True)
    dev_data_tgt = read_corpus(
        args['--dev-tgt'], source='tgt', dev_mode=dev_mode, space_tokenize=True)

    if do_bleu:
        test_data_src = read_corpus(
            args['--test-src'], source='src', dev_mode=dev_mode
        )
        test_data_tgt = read_corpus(
            args['--test-tgt'], source='tgt', dev_mode=dev_mode
        )

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    
    print (train_data[0][0])
    print (train_data[0][1])

    max_tokens_in_sentence = int(args['--max-decoding-time-step'])
    train_data = clean_data(train_data, max_tokens_in_sentence)
    dev_data = clean_data(dev_data, max_tokens_in_sentence)

    train_batch_size = int(args['--batch-size'])
    dev_batch_size = 128
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    bleu_niter = int(args['--bleu-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']

    vocab = Vocab.load(args['--vocab'], args['--word_freq'])

    # model = NMT(embed_size=int(args['--embed-size']),
    #             hidden_size=int(args['--hidden-size']),
    #             dropout_rate=float(args['--dropout']),
    #             vocab=vocab)
    # writer = SummaryWriter()

    # model = TransformerNMT(vocab, num_hidden_layers=3)

    #model.train()

    # uniform_init = float(args['--uniform-init'])
    # if np.abs(uniform_init) > 0.:
    #     print('uniformly initialize parameters [-%f, +%f]' %
    #           (uniform_init, uniform_init), file=sys.stderr)
    #     for p in model.parameters():
    #         if p.dim() > 1:
    #             torch.nn.init.xavier_uniform_(p)
    #         else:
    #             p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    # device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    # print('use device: %s' % device, file=sys.stderr)

    # model = model.to(device)

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    print("Sorting dataset based on difficulty...")
    dataset = (train_data, dev_data)
    ordered_dataset = load_order(args['--order-name'], dataset, vocab)
    # TODO: order = balance_order(order, dataset)
    (train_data, dev_data) = ordered_dataset
    
    write_to_file(train_data)

    visualize_scoring_examples = False
    if visualize_scoring_examples:
        visualize_scoring(ordered_dataset, vocab)
        
    import sys
    print ("Stored the exemplars")
    sys.exit(0)

    n_iters = math.ceil(len(train_data) / train_batch_size)
    print("n_iters per epoch is {}: ({} / {})".format(n_iters,
                                                      len(train_data), train_batch_size))
    max_epoch = int(args['--max-epoch'])
    max_iters = max_epoch * n_iters

    print('begin Maximum Likelihood training')
    print('Using order function: {}'.format(args['--order-name']))
    print('Using pacing function: {}'.format(args['--pacing-name']))
    while True:
        epoch += 1
        for _ in range(n_iters):
            # Get pacing data according to train_iter
            current_train_data, current_dev_data = pacing_data(
                train_data, 
                dev_data, 
                time=train_iter, 
                warmup_iters=int(args["--warmup-iters"]),
                method=args['--pacing-name'],
                tb=writer
            )

            # Uniformly sample batches from the paced dataset
            src_sents, tgt_sents = get_pacing_batch(
                current_train_data,
                batch_size=train_batch_size,
                shuffle=True
            )

            train_iter += 1

            # ERROR START
            optimizer.zero_grad()

            batch_size = len(src_sents)

            example_losses = -model(src_sents, tgt_sents)  # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val: int = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(
                len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec'
                      % (epoch, train_iter,
                         report_loss / report_examples,
                         math.exp(
                             report_loss / report_tgt_words),
                         cum_examples,
                         report_tgt_words /
                         (time.time(
                         ) - train_time),
                         time.time() - begin_time), file=sys.stderr)
                writer.add_scalar('Loss/train', report_loss /
                                  report_examples, train_iter)
                writer.add_scalar('ppl/train', math.exp(
                    report_loss / report_tgt_words), train_iter)
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # evaluate BLEU
            if train_iter % bleu_niter == 0 and do_bleu:
                bleu = decode_with_params(model,
                    test_data_src,
                    test_data_tgt,
                    int(args['--beam-size']),
                    int(args['--max-decoding-time-step'])
                    )
                writer.add_scalar('bleu/test', bleu, train_iter)

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d'
                      % (epoch, train_iter,
                         cum_loss / cum_examples,
                         np.exp(
                             cum_loss / cum_tgt_words),
                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = evaluate_ppl(
                    model, current_dev_data, batch_size=dev_batch_size)
                valid_metric = -dev_ppl
                writer.add_scalar('ppl/valid', dev_ppl, train_iter)
                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('validation: iter %d, dev. ppl %f' %
                      (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(
                    hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print(
                        'save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(),
                               model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * \
                            float(args['--lr-decay'])
                        print(
                            'load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(
                            model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers',
                              file=sys.stderr)
                        optimizer.load_state_dict(
                            torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch >= int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    print("load test source sentences from [{}]".format(
        args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(
            args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')
    else:
        test_data_tgt = None

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    beam_size = int(args['--beam-size'])
    max_decoding_time_step = int(args['--max-decoding-time-step'])
    output_file = args['OUTPUT_FILE']

    decode_with_params(
        model, test_data_src, test_data_tgt, beam_size,
        max_decoding_time_step, output_file)


def decode_with_params(
        model,
        test_data_src,
        test_data_tgt,
        beam_size,
        max_decoding_time_step,
        output_file: str = None):
    hypotheses = beam_search(model, test_data_src,
                             beam_size=beam_size,
                             max_decoding_time_step=max_decoding_time_step)

    if test_data_tgt:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(
            test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    if output_file:
        with open(output_file, 'w') as f:
            for src_sent, hyps in zip(test_data_src, hypotheses):
                top_hyp = hyps[0]
                hyp_sent = ' '.join(top_hyp.value)
                f.write(hyp_sent + '\n')
    return bleu_score * 100


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(
                src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training:
        model.train(was_training)

    return hypotheses


def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # Check pytorch version
    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(
        torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
