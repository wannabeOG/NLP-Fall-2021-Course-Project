# @author: Kartheek Akella (CVIT)

from fairseq import options, utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.fairseq_model import CVITLanguageModel

from fairseq.models.transformer import (
    Embedding,
    TransformerEncoder,
)
from fairseq.modules import (
    AdaptiveInput,
    CharacterTokenEmbedder,
)
from torch.nn import Linear

DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('encoder_lm')
class TransformerLanguageModel(CVITLanguageModel):

    def __init__(self, encoder, out_layer):
        super().__init__(encoder, out_layer)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-output-dim', type=int, metavar='N',
                            help='encoder output dimension')
        parser.add_argument('--encoder-input-dim', type=int, metavar='N',
                            help='encoder input dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--no-encoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last encoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-encoder-input-output-embed', action='store_true',
                            help='share encoder input and output embeddings')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', default=2, type=int, metavar='N',
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D',
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep',
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D',
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D',
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D',
                            help='scalar quantization noise and scalar quantization at training time')

        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = getattr(args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary, eval(args.character_filters),
                args.character_embedding_dim, args.encoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary), task.source_dictionary.pad(), args.encoder_input_dim,
                args.adaptive_input_factor, args.encoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq, args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(args, task.source_dictionary, args.encoder_input_dim)

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.encoder_input_dim == args.encoder_output_dim

        encoder = TransformerEncoder(
            args, task.target_dictionary, embed_tokens,
        )
        print('Encoder Output Dimensions:', args.encoder_output_dim)
        print('Output Size:',len(task.target_dictionary))
        linear_layer = Linear(args.encoder_output_dim, len(task.target_dictionary))
        return cls(encoder, linear_layer)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens


@register_model_architecture('encoder_lm', 'encoder_lm')
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, 'no_tie_adaptive_proj'):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_encoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, 'encoder_final_norm'):
        args.no_encoder_final_norm = not args.encoder_final_norm

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)

    args.max_source_positions = getattr(args, 'max_source_positions', 1024)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.encoder_layerdrop = getattr(args, 'encoder_layerdrop', 0)
    args.encoder_layers_to_keep = getattr(args, 'encoder_layers_to_keep', None)
    args.quant_noise_pq = getattr(args, 'quant_noise_pq', 0)
    args.quant_noise_pq_block_size = getattr(args, 'quant_noise_pq_block_size', 8)
    args.quant_noise_scalar = getattr(args, 'quant_noise_scalar', 0)

    args.add_bos_token = getattr(args, 'add_bos_token', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', False)
    args.character_embeddings = getattr(args, 'character_embeddings', False)

    args.encoder_output_dim = getattr(args, 'encoder_output_dim', args.encoder_embed_dim)
    args.encoder_input_dim = getattr(args, 'encoder_input_dim', args.encoder_embed_dim)

    # Model training is not stable without this
    args.encoder_normalize_before = True
    args.no_encoder_final_norm = getattr(args, 'no_encoder_final_norm', False)

    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)

    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)
    