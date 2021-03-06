#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./data/iwslt/fr-en/train.fr-en.fr --train-tgt=./data/iwslt/fr-en/train.fr-en.en --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en --test-src=./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.fr --test-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.en --vocab=iwslt_vocab.json --word_freq=iwslt_word_freq.json --cuda
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.fr ./data/iwslt/fr-en/IWSLT16.TED.tst2014.fr-en.en outputs/test_outputs.txt --cuda
elif [ "$1" = "dev" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr ./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en outputs/test_outputs_dev.txt --cuda
elif [ "$1" = "train_local" ]; then
	python run.py train --train-src=./data/iwslt/fr-en/train.fr-en.fr --train-tgt=./data/iwslt/fr-en/train.fr-en.en --dev-src=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.fr --dev-tgt=./data/iwslt/fr-en/IWSLT16.TED.tst2013.fr-en.en --vocab=iwslt_vocab.json --word_freq=iwslt_word_freq.json --order-name $2 --pacing-name $3 --ignore-test-bleu 1
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt
elif [ "$1" = "download" ]; then
	python -c $"from torchnlp.datasets import iwslt_dataset
train, dev, test = iwslt_dataset(language_extensions=['fr', 'en'], train=True, dev=True, test=True)"
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=/content/drive/My\ Drive/RA-Project-IIIT-H/Cat_Forgetting/Datasets/en-fr/train.fr-en.fr --train-tgt=/content/drive/My\ Drive/RA-Project-IIIT-H/Cat_Forgetting/Datasets/en-fr/train.fr-en.en europarl_vocab_normal.json europarl_word_freq_normal.json --freq-cutoff 5
elif [ "$1" = "rearrange_datasets" ]; then
	python run.py train --train-src=/content/drive/My\ Drive/RA-Project-IIIT-H/Cat_Forgetting/Datasets/en-fr/train.fr-en.fr --train-tgt=/content/drive/My\ Drive/RA-Project-IIIT-H/Cat_Forgetting/Datasets/en-fr/train.fr-en.en --dev-src=/content/drive/My\ Drive/RA-Project-IIIT-H/Cat_Forgetting/Datasets/en-fr/dev.fr-en.fr --dev-tgt=/content/drive/My\ Drive/RA-Project-IIIT-H/Cat_Forgetting/Datasets/en-fr/dev.fr-en.en --vocab=europarl_vocab_normal.json --word_freq=europarl_word_freq_normal.json --order-name $2 --pacing-name $3 --ignore-test-bleu 1
else
	echo "Invalid Option Selected"
fi
