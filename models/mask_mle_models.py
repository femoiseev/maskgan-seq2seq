import numpy as np
from copy import deepcopy
import torch
import fairseq
from fairseq.models import FairseqEncoder
from fairseq.models.transformer import TransformerEncoder
import torch.nn as nn


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
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
           
#         src_mask = src_tokens == self.src_dictionary.index(sym=self.SPLIT_SYMBOL)
#         print(self.src_dictionary.string(src_tokens[:, 1]))
#         print(src_mask[0])
#         print(src_mask[1])
#         print(src_mask[2])
#         print(src_mask[3])
#         print("*" * 30)
#         for i in range(128):
#             print(self.src_dictionary.index(src_tokes[0][i], endline=" "))

#         print("*" * 40)
#         print("src")
#         print(self.src_dictionary.string(src_tokens))
#         print("tgt")
#         print(self.dst_dictionary.string(masked_tgt))
#         print("*" * 40)
        
        src_dict = self.encoder(src_tokens, src_lengths)
#         print("usual encoder ok")

#         print("*" * 40)
#         print(torch.max(src_lengths))
#         print(tgt_lengths.size())
#         print(torch.max(tgt_lengths))
#         print("*" * 40)
        dst_dict = self.masked_encoder(masked_tgt, tgt_lengths)
#         print("masked ok")
#         print("masked tgt")
#         print(masked_tgt.size())
#         print("src ")
#         print(src_tokens.size())
#         print(torch.cat([src_dict['encoder_out'], dst_dict['encoder_out']], dim=-1).size())
        
        res = {
            'encoder_out': torch.cat([src_dict['encoder_out'], dst_dict['encoder_out']], dim=-1),
            'encoder_padding_mask': None
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
        
        
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        return encoder_out