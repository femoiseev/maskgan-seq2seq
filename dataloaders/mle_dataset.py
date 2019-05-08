import torch
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.data import data_utils
    
    
class MLELanguagePairDataset(LanguagePairDataset):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cnts = 0

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch with the following keys:
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        if len(samples) == 0:
            return {}

        def merge(key, pad, eos, left_pad, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad, eos, left_pad, move_eos_to_beginning,
            )

        id = torch.LongTensor([s['id'] for s in samples])
        src_tokens = merge('source', self.src_dict.pad(),
                           self.src_dict.eos(),
                           left_pad=self.left_pad_source)
        
        # sort by descending source length
        src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
        src_lengths, sort_order = src_lengths.sort(descending=True)
        id = id.index_select(0, sort_order)
        src_tokens = src_tokens.index_select(0, sort_order)

        prev_output_tokens = None
        target = None
        tgt_lengths = None
        ok_target = None
        if samples[0].get('target', None) is not None:
            
            target = merge('target', self.tgt_dict.pad(),
                           self.tgt_dict.eos(),
                           left_pad=self.left_pad_source)
            ok_target = merge('target', self.tgt_dict.pad(),
                              self.tgt_dict.eos(),
                              left_pad=self.left_pad_target)

            tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples])
            target = target.index_select(0, sort_order)
            ok_target = ok_target.index_select(0, sort_order)
            tgt_lengths = tgt_lengths.index_select(0, sort_order)
            ntokens = sum(len(s['target']) for s in samples)

            if self.input_feeding:
                # we create a shifted version of targets for feeding the
                # previous output token(s) into the next decoder step
                prev_output_tokens = merge(
                    'target', self.tgt_dict.pad(),
                    self.tgt_dict.eos(),
                    left_pad=self.left_pad_target,
                    move_eos_to_beginning=True,
                )
                prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        else:
            ntokens = sum(len(s['source']) for s in samples)

        p = 0.5
        mask = torch.distributions.Bernoulli(torch.Tensor([p]))
        mask_tensor = None

        if samples[0].get('target', None) is not None:
            mask_tensor = mask.sample(target.size())[:, :, 0]

            # target[mask_tensor.byte()] = self.tgt_dict.index("<MASK>")
            target[(target != self.tgt_dict.pad()) & (mask_tensor.byte())] = self.tgt_dict.index("<MASK>")
            mask_tensor[(target == self.tgt_dict.pad())] = 0
            # for i in range(len(target)):
            #     for j in range(len(target[i])):
            #         if target[i, j] != self.tgt_dict.pad():
            #             mask_val = mask.sample()
            #             if mask_val:
            #                 target[i, j] = self.tgt_dict.index("<MASK>")

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
                'masked_tgt': target,
                'tgt_lengths': tgt_lengths,
            },
            'target': ok_target,
            'masks': mask_tensor
        }
        self.cnts += 1

        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens
        return batch
