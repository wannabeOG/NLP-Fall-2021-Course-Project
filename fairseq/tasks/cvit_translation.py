# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    # data_utils,
    # indexed_dataset,
    LanguagePairDataset,
)

from fairseq.data.cvit.dataset import CVITIndexedRawTextDataset

from . import FairseqTask, register_task


def load_langpair_dataset(
    corpus_pairs, dictionary, tokenizer, combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions, max_target_positions,
):
    # Load everything from config file
    # print(corpus_pairs)
    src_datasets, tgt_datasets = [], []
    for pair in corpus_pairs:
        src, tgt = pair

        def wrapped(left, right):
            left_dataset = CVITIndexedRawTextDataset(left, tokenizer, dictionary, tgt_lang=right.lang)
            right_dataset = CVITIndexedRawTextDataset(right, tokenizer, dictionary)
            return left_dataset, right_dataset

        # tgt -> src
        x, y = wrapped(src, tgt)
        src_datasets.append(x)
        tgt_datasets.append(y)

        # src -> tgt
        # x, y = wrapped(tgt, src)
        # src_datasets.append(x)
        # tgt_datasets.append(y)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]

    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    return LanguagePairDataset(
        src_dataset, src_dataset.sizes, dictionary,
        tgt_dataset, tgt_dataset.sizes, dictionary,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
    )

@register_task('shared-multilingual-translation')
class CVITTranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, data):
        super().__init__(args)
        self.data = data
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, flag, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        # paths = args.data.split(':')
        # assert len(paths) > 0
        # # find language pair automatically
        # if args.source_lang is None or args.target_lang is None:
        #     args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        # if args.source_lang is None or args.target_lang is None:
        #     raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        # print(args.data)
        def read_config(path):
            with open(path) as config:
                import yaml
                contents = config.read()
                data = yaml.load(contents)
                return data

        path = "/content/drive/My Drive/RA-Project-IIIT-H/Cat_Forgetting/fairseq-working/config.yaml"
        data = read_config(path)
        # self.pairs = pairs_select(data['corpo
        # ra'])

        # src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        # tgt_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        
        if (flag == 1):
            src_dict = cls.load_dictionary(data['dictionary_comb']['src'])
            tgt_dict = cls.load_dictionary(data['dictionary_comb']['tgt'])
        
        elif (flag == 2):
            src_dict = cls.load_dictionary(data['dictionary_og']['src'])
            tgt_dict = cls.load_dictionary(data['dictionary_og']['tgt'])
            
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        assert src_dict == tgt_dict
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict, data)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # paths = self.args.data.split(':')
        # assert len(paths) > 0
        # data_path = paths[epoch % len(paths)]
        from fairseq.data.cvit.utils import pairs_select
        pairs = pairs_select(self.data['corpora'], split)

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        from ilmulti.sentencepiece import SentencePieceTokenizer
        tokenizer = SentencePieceTokenizer(self.data['hard_coded_dict'])

        # tokenizer test
        tokens = tokenizer("Hello World !", lang = 'en')

        self.datasets[split] = load_langpair_dataset(pairs, self.src_dict, tokenizer,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
