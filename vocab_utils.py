import sentencepiece as spm
from ilmulti.sentencepiece import SentencePieceTokenizer

#path to training data is required
#input -> input to the corpus 
#model_prefix -> "model_prefix.model" "model_prefix.vocab" will be generated. We will need both these
#vocab size that we will need for the task
#model_type unigram (default), bpe, char, or word (must be pre-tokenized)
#character_coverage 0.995 for large character set and 1 for small character set
#user_defined_symbols -> may not be necessary tbh

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)

    # Optional
    parser.add_argument('--src', type=str)
    parser.add_argument('--tgt', type=str)
    parser.add_argument('--single', action='store_true')


    args = parser.parse_args()

def generate_vocab():


spm.SentencePieceTrainer.train(
   input = training_dict['ml'], 
   model_prefix = model_tags['ml'], 
   vocab_size = config['ml'], 
   character_coverage = 0.9995, 
   model_type = 'unigram' 
)


def _get_vocabulary_mapping(current_vocab_path, new_vocab_path, mode):
    """
    Maps vocabulary new indices to old ones. -1 means that the entry is new.
    """
    current_vocab = Vocab(from_file=current_vocab_path)
    new_vocab = Vocab(from_file=new_vocab_path)
    mapping = []

    #Directly combines both the vocabularies into one vocab
    if mode == "merge":

        final_vocab = Vocab(from_file=current_vocab_path)
        mapping = [i for i in range(current_vocab.size)]
        #Iterate over the new vocab
        for new_word in new_vocab.words:
            #Check if the word from the new vocab exists in the current vocab
            if current_vocab.lookup(new_word) is None:
                #mapping = [1, 2, 3, 4, 5, -1, -1]
                mapping.append(-1)
                final_vocab.add(new_word)

    #Init the final vocab from new vocab 
    elif mode == "replace":
        final_vocab = new_vocab
        for new_word in new_vocab.words:
            idx = current_vocab.lookup(new_word)
            #new words are looked in the current vocab, the idx is stored in the mapping
            if idx is not None:
                mapping.append(idx)
            #Word in the new vocab does not exist in current vocab, append -1 now 
            else:
                mapping.append(-1)
    #mapping = [1, 2, 3, 4, 5, -1, -1, 5] for merge 
    #mapping = [212, 11, 27, 221, -1, -1, 4] for replace            
    mapping.append(current_vocab.size)  # <unk> token is always the last entry.
    return mapping, final_vocab

def _update_vocabulary_variable(variable, vocab_size, mapping, init):
    """
    Creates a new variable, possibly copying previous entries based on mapping.
    """
    
    dim = variable.shape.index(vocab_size)
    # Make the dimension to index the first.
    perm = list(range(len(variable.shape)))
    perm[0], perm[dim] = perm[dim], perm[0]
    variable_t = np.transpose(variable, axes=perm)
    new_shape = list(variable_t.shape)
    new_shape[0] = len(mapping)

    if init == "zeros":
        new_variable_t = np.zeros(new_shape, dtype=variable.dtype)
    elif init == "random":
        new_variable_t = np.random.uniform(low=-0.99, high=0.99, size=new_shape).astype(variable.dtype)

    for i, j in enumerate(mapping):
        if j >= 0:
            new_variable_t[i] = variable_t[j]
    
    new_variable = np.transpose(new_variable_t, axes=perm)
    return new_variable