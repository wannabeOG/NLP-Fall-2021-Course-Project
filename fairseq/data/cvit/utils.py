from collections import defaultdict
from itertools import combinations
from . import DATASET_REGISTRY

def canonicalize(langcode):
    _variations = {
        "ur": ["ur", "ud"],
        "bn": ["bg", "bn"]
    }

    inverse = {}
    for root in _variations:
        for x in _variations[root]:
            inverse[x] = root

    return inverse.get(langcode, langcode)

def select(tags, splits, langs):
    """
    """
    # Filter by split, langs
    registry = dict([
            (k, v)  \
            for k, v in DATASET_REGISTRY.items() \
            if k in tags
    ])

    filtered_corpora = []
    for key in registry:
        _splits, f = registry[key]
        isplits = set(_splits).intersection(set(splits))
        isplits = list(isplits)
        for _split in isplits:
            corpora = f(_split)
            corpora = [
                c for c in corpora \
                if c.lang in langs
            ]

            filtered_corpora.extend(corpora)


    def group_by_tag(corpora):
        _dict = defaultdict(list)
        for corpus in corpora:
            _dict[corpus.tag].append(corpus)
        return _dict

    corpora = group_by_tag(filtered_corpora)
    pairs = []
    for key in corpora:
        # TODO(jerin): Sort for determinism
        for dx, dy in combinations(corpora[key], 2):
            pairs.append((dx, dy))


    return pairs


def pairs_select(corpora_config, split):
    ls = []
    if split == 'valid': split = 'dev'
    for tag, v in corpora_config.items():
        tags = [tag]
        if split in v['splits']:
            splits = [split]
            pairs = select(tags, splits, v['langs'])
            ls.extend(pairs)

    # Set is non-determinism. Sort
    def sort_key(pair):
        first, second = pair
        return (first.path, second.path)


    unique = list(set(ls))
    unique = sorted(unique, key=sort_key)
    return unique

###################################### MONO PRETRAIN CODE ###########################################
def pretrain_select(tags, splits, langs):
    """
    """
    #TODO: Add Documentation and Comments
    registry = dict([
            (k, v)  \
            for k, v in DATASET_REGISTRY.items() \
            if k in tags
    ])

    filtered_corpora = []
    for key in registry:
        _splits, f = registry[key]
        isplits = set(_splits).intersection(set(splits))
        isplits = list(isplits)
        for _split in isplits:
            corpora = f(_split)

            corpora = [
                c for c in corpora \
                if c.lang in langs
            ]

            filtered_corpora.extend(corpora)


    def group_by_tag(corpora):
        _dict = defaultdict(list)
        for corpus in corpora:
            _dict[corpus.tag].append(corpus)
        return _dict

    corpora = group_by_tag(filtered_corpora)
    pairs = []

    #TODO: Is this loop necessary
    for key in corpora:
        print ("This is the key", key)
        print ("Corpora key", corpora[key])
    
    return corpora[key]

def monoling_select(corpora_config, split):
    """
    """
    #TODO: Add Documentation and Comments
    ls = []
    if split == 'valid': split = 'dev'
    for tag, v in corpora_config.items():
        tags = [tag]
        if split in v['splits']:
            splits = [split]
            pairs = pretrain_select(tags, splits, v['langs'])
            ls = pairs

    # Set is non-determinism. Sort
    def sort_key(pair):
        first, second = pair
        return (first.path, second.path)

    unique = list(set(ls))
    return unique[0]
#######################################################################################################



if __name__ == '__main__':
    tags = ['iitb-hi-en', 'wat-ilmpc']
    splits = ['train']
    langs = ['en', 'hi', 'ta', 'ml']
    pairs = select(tags, splits, langs)
    from pprint import pprint
    pprint(pairs)

