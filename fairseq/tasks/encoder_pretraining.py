import os

import torch

from fairseq import utils
from fairseq.data import (
    data_utils,
    Dictionary,
    MonolingualDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)
from fairseq.tasks import FairseqTask, register_task

from fairseq.data.cvit.dataset import CVITIndexedRawTextDataset


@register_task('encoder_pretraining')
class CVITPretrainingTask(FairseqTask):
    """
    Train a language model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the language model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the language model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The language modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The language modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.language_modeling_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=1024, type=int,
                            help='max number of tokens per sample for LM dataset')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--output-dictionary-size', default=-1, type=int,
                            help='limit the size of output dictionary')
        parser.add_argument('--self-target', action='store_true',
                            help='include self target')
        parser.add_argument('--future-target', action='store_true',
                            help='include future target')
        parser.add_argument('--past-target', action='store_true',
                            help='include past target')
        parser.add_argument('--add-bos-token', action='store_true',
                            help='prepend beginning of sentence token (<s>)')
        parser.add_argument('--max-target-positions', type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        # fmt: on

    def __init__(self, args, data, dictionary, output_dictionary=None, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary or dictionary
        self.data = data

        if targets is None:
            targets = ['future']
        self.targets = targets

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        print ("These are the arguments", args)
        print ("\n")

        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        dictionary = None
        output_dictionary = None

        def read_config(path):
            with open(path) as config:
                import yaml
                contents = config.read()
                data = yaml.load(contents)
                return data

        # if args.data:
        #     paths = args.data.split(':')
        #     assert len(paths) > 0
        #     dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
        #     print('| dictionary: {} types'.format(len(dictionary)))
        #     output_dictionary = dictionary
        #     if args.output_dictionary_size >= 0:
        #         output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)

        ######################### CHANGE THIS #################################################
        args.data = read_config("/home2/asvs/fairseq-working/config.yaml")
        ########################################################################################
        if (args.data):
            path = args.data['dictionary']['src']
            dictionary = Dictionary.load(path)
            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = dictionary
            if (args.output_dictionary_size >= 0):
                output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)
        
        # upgrade old checkpoints
        if hasattr(args, 'exclude_self_target'):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, 'self_target', False):
            targets.append('self')
        if getattr(args, 'future_target', False):
            targets.append('future')
        if getattr(args, 'past_target', False):
            targets.append('past')
        if len(targets) == 0:
            # standard language modeling
            targets = ['future']

        return cls(args, args.data, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)
        
        #print ("This is the model", model)
        #print ("This is the model dict", model.state_dict().keys())

        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError('Unsupported language modeling target: {}'.format(target))

        return model

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        print ("This is the split", split)

        from fairseq.data.cvit.utils import monoling_select
        dataset = monoling_select(self.data['corpora'], split)
        
        from ilmulti.sentencepiece import SentencePieceTokenizer

        hard_code_dict = self.data['hard_coded_dict']
        
        tokenizer = SentencePieceTokenizer(hard_code_dict)
        dataset = CVITIndexedRawTextDataset(dataset, tokenizer, self.dictionary)
        
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        dataset = TokenBlockDataset(
            dataset, dataset.sizes, self.args.tokens_per_sample,
            pad=self.dictionary.pad(), eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode, include_targets=True,
        )

        add_eos_for_other_targets = self.args.sample_break_mode is not None and self.args.sample_break_mode != 'none'

        self.datasets[split] = MonolingualDataset(
            dataset, dataset.sizes, self.dictionary, self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets, shuffle=True,
            targets=self.targets, add_bos_token=self.args.add_bos_token,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return TransformEosDataset(
            MonolingualDataset(
                TokenBlockDataset(
                    src_tokens,
                    src_lengths,
                    block_size=None,
                    pad=self.source_dictionary.pad(),
                    eos=self.source_dictionary.eos(),
                    break_mode='eos',
                    include_targets=False,
                ),
                src_lengths,
                self.source_dictionary,
                self.target_dictionary,
                add_eos_for_other_targets=False,
                shuffle=False,
                add_bos_token=self.args.add_bos_token,
            ),
            eos=self.source_dictionary.eos(),
            # remove EOS since this will be used as a prefix for generation
            remove_eos_from_src=True,
            has_target=False,
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None and sample['net_input']['src_tokens'].nelement():
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample['net_input']['src_tokens']
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
        