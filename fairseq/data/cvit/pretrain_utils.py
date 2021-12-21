from fairseq import checkpoint_utils
from collections import OrderedDict
import os 

def init_tmodel(source_path, target_path, modified_path):
    """
    Args:
        source_path: A fairseq.Language_model that whose params will be initialized with the params
                    from the Transformer model.
        target_path: A fairseq.Transformer model that has been trained on the Translation task
        modified_path: A string object denoting the path to where you wish to store the model
    """
    encoder_state = checkpoint_utils.load_checkpoint_to_cpu(source_path)
    translation_state = checkpoint_utils.load_checkpoint_to_cpu(target_path)

    filtered_state = []

    for key in encoder_state['model'].keys():
        filtered_state.append((key, encoder_state['model'][key]))

    #Remove the linear and layer norm layers to maintain compatibiility
    filtered_state.pop()
    filtered_state.pop()
    filtered_state.pop()
    filtered_state.pop()

    list_translation_state = []

    for key in translation_state['model'].keys():
        list_translation_state.append((key, translation_state['model'][key]))

    for index, key in enumerate(list_translation_state):
      if key[0].startswith('encoder'):
        list_translation_state[index] = filtered_state[index]

    list_translation_state_dict = OrderedDict(list_translation_state)
    translation_state['model'] = list_translation_state_dict
    
    checkpoint_utils.torch_persistent_save(translation_state, modified_path)

    return 

def split_create(model, 
                source_path = "/home/wannabe/Documents/ufal-transformer-big/transformer_checkpoints/checkpoint_last.pt", 
                target_path = "/home/wannabe/Documents/ufal-transformer-big/encoder_checkpoints/checkpoint_last.pt"
            ):
    """
    Args:
        source_path: A fairseq.Language_model that whose params will be initialized with the params
                    from the Transformer model.
        target_path: A fairseq.Transformer model that has been trained on the Translation task
        modified_path: A string object denoting the path to where you wish to store the model
    """

    #check if the file exists, it it does return
    if os.path.isfile(target_path):
        #print ("Inside the if clause")
        return 

    extended_list = []
    for key in model.state_dict().keys():
        if key.startswith('encoder.layer_norm') or key.startswith('out_layer'):
            extended_list.append((key, model.state_dict()[key]))

    translation_state = checkpoint_utils.load_checkpoint_to_cpu(source_path)

    #filtered state has the encoder parts of the translation model
    filtered_state = []

    for key in translation_state['model'].keys():
        if key.startswith('encoder'):
            filtered_state.append((key, translation_state['model'][key]))

    filtered_state.extend(extended_list)

    list_encoder_state_dict = OrderedDict(filtered_state)
    translation_state['model'] = list_encoder_state_dict

    #save the encoder weights of the translation model at the target path 
    checkpoint_utils.torch_persistent_save(translation_state, target_path)

    return 


if __name__ == '__main__':
    #parser = ArgumentParser()
    #parser.add_argument('config_file', help='config file')
    #parser.add_argument('--rebuild', action='store_true')
    
    #args = parser.parse_args()
    source_path = "/home/wannabe/Documents/ufal-transformer-big/transformer_checkpoints/checkpoint_last.pt"
    target_path = "/home/wannabe/Documents/ufal-transformer-big/encoder_checkpoints/checkpoint_last.pt"

    #split_create(source_path, target_path)
    encoder_state = checkpoint_utils.load_checkpoint_to_cpu(target_path)
    
    for key in encoder_state['model'].keys():
        print (key)
    
