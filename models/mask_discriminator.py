import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import FairseqModel, register_model
from fairseq.models.transformer import Embedding

from .mask_layers import (
    MaskTransformerDecoder,
    MaskTransformerEncoder
)


class MaskDiscriminatorDecoder(MaskTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, left_pad, final_norm)

        output_embed_dim = args.decoder_output_dim
        self.logits = nn.Linear(output_embed_dim, 1)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn_source, attn_mask = None, None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn_source, attn_mask = layer(
                x,
                encoder_out['source_encoder_out'] if encoder_out is not None else None,
                encoder_out['source_encoder_padding_mask'] if encoder_out is not None else None,
                encoder_out['mask_encoder_out'] if encoder_out is not None else None,
                encoder_out['mask_encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.normalize:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        x = self.logits(x)
        return x, {'attn': attn_source, 'inner_states': inner_states}


@register_model('mask_discriminator')
class MaskTransformerDiscriminator(FairseqModel):

    def __init__(self, encoder, decoder):
        print("created MLE Discriminator")
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        #         # make sure all arguments are present in older models
        #         base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):

            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        src_embed_tokens = build_embedding(
            src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )

        tgt_embed_tokens = build_embedding(
            tgt_dict, args.encoder_embed_dim, args.decoder_embed_path
        )

        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
        )

        encoder = MaskTransformerEncoder(args,
                                         src_dict, tgt_dict,
                                         src_embed_tokens, tgt_embed_tokens)

        decoder = MaskDiscriminatorDecoder(args, tgt_dict, decoder_embed_tokens)

        return MaskTransformerDiscriminator(encoder, decoder)

    def forward(self, src_tokens, src_lengths, masked_tgt, tgt_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths, masked_tgt, tgt_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out)

        return decoder_out
