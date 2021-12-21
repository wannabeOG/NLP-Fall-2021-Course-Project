from argparse import ArgumentParser
from fairseq.data.cvit.utils import pairs_select
from fairseq.data.cvit.dataset import _CVITIndexedRawTextDataset
from fairseq.data.cvit.lmdb import LMDBCorpusWriter, LMDBCorpus
import yaml
from multiprocessing import Pool
from functools import partial
import os
from ilmulti.sentencepiece import SentencePieceTokenizer

