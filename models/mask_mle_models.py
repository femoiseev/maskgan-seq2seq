import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules import (
    AdaptiveSoftmax, LearnedPositionalEmbedding, SinusoidalPositionalEmbedding, MultiheadAttention
)


class MLETransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
        left_pad (bool, optional): whether the input is left-padded
            (default: True).
    """

    def __init__(self, args, src_dictionary, dst_dictionary, 
                 src_embed_tokens, dst_embed_tokens, left_pad=True):
        
        super().__init__(None)
            
        self.src_dictionary = src_dictionary
        self.dst_dictionary = dst_dictionary
        self.encoder = TransformerEncoder(args, src_dictionary,
                                          src_embed_tokens, left_pad=left_pad)
        
        self.masked_encoder = TransformerEncoder(args, dst_dictionary,
                                                 dst_embed_tokens, left_pad=left_pad)

    def forward(self, src_tokens, src_lengths, masked_tgt, tgt_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            masked_tgt (LongTensor): masked target (batch, tgt_len)
            tgt_lengths (torch.LongTensor): length of target, (batch)
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions

        src_dict = self.encoder(src_tokens, src_lengths)
        dst_dict = self.masked_encoder(masked_tgt, tgt_lengths)

        res = {
            'source_encoder_out': src_dict['encoder_out'],
            'source_encoder_padding_mask': src_dict['encoder_padding_mask'],
            'mask_encoder_out': dst_dict['encoder_out'],
            'mask_encoder_padding_mask': dst_dict['encoder_padding_mask']
        }
        return res

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """

        if encoder_out['source_encoder_out'] is not None:
            encoder_out['source_encoder_out'] = \
                encoder_out['source_encoder_out'].index_select(1, new_order)
        if encoder_out['source_encoder_padding_mask'] is not None:
            encoder_out['source_encoder_padding_mask'] = \
                encoder_out['source_encoder_padding_mask'].index_select(0, new_order)

        if encoder_out['mask_encoder_out'] is not None:
            encoder_out['mask_encoder_out'] = \
                encoder_out['mask_encoder_out'].index_select(1, new_order)
        if encoder_out['mask_encoder_padding_mask'] is not None:
            encoder_out['mask_encoder_padding_mask'] = \
                encoder_out['mask_encoder_padding_mask'].index_select(0, new_order)
        return encoder_out


class MLETransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
        super().__init__(dictionary)
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            left_pad=left_pad,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            MaskDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, output_embed_dim, bias=False) \
            if embed_dim != output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=output_embed_dim ** -0.5)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = args.decoder_normalize_before and final_norm
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

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
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)
        return x, {'attn': attn_source, 'inner_states': inner_states}

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]
        if utils.item(state_dict.get('{}.version'.format(name), torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict['{}.version'.format(name)] = torch.Tensor([1])

        return state_dict


class MaskDecoderLayer(nn.Module):

    def __init__(self, args, no_encoder_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if no_encoder_attn:
            self.source_encoder_attn = None
            self.mask_encoder_attn = None
            self.encoder_attn_layer_norm = None
            self.concat_dense = None
        else:
            self.source_encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.mask_encoder_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
            self.concat_dense = Linear(2 * self.embed_dim, self.embed_dim, bias=True)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, source_encoder_out, source_encoder_padding_mask,
                mask_encoder_out, mask_encoder_padding_mask, incremental_state,
                prev_self_attn_state=None, prev_source_attn_state=None, prev_mask_attn_state=None,
                self_attn_mask=None, self_attn_padding_mask=None):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        attn_source = None
        attn_mask = None
        if self.source_encoder_attn is not None:
            residual = x
            source_x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
            mask_x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)

            self.set_attention_input_buffer(self.source_encoder_attn, incremental_state, prev_source_attn_state)
            self.set_attention_input_buffer(self.mask_encoder_attn, incremental_state, prev_mask_attn_state)

            source_x, attn_source = self.source_encoder_attn(
                query=source_x,
                key=source_encoder_out,
                value=source_encoder_out,
                key_padding_mask=source_encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )

            mask_x, attn_mask = self.mask_encoder_attn(
                query=mask_x,
                key=mask_encoder_out,
                value=mask_encoder_out,
                key_padding_mask=mask_encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=(not self.training and self.need_attn),
            )
            x = torch.cat([source_x, mask_x], dim=-1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.concat_dense(x))
            x = residual + x
            x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn_source, attn_mask, self_attn_state
        return x, attn_source, attn_mask

    def set_attention_input_buffer(self, attention_layer, incremental_state, previous_attn_state):
        if previous_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = previous_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            attention_layer._set_input_buffer(incremental_state, saved_state)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad, learned=False):
    if learned:
        m = LearnedPositionalEmbedding(num_embeddings + padding_idx + 1, embedding_dim, padding_idx, left_pad)
        nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
        nn.init.constant_(m.weight[padding_idx], 0)
    else:
        m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, num_embeddings + padding_idx + 1)
    return m
