import math
import torch
import torch.nn.functional as F
from copy import deepcopy

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('discriminator_loss')
class DiscriminatorCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        real_output = model(**sample['net_input'])[0]
        fake_input = deepcopy(sample['net_input'])
        fake_input['prev_output_tokens'] = sample['generated_sents']
        fake_output = model(**fake_input)[0]

        loss, _ = self.compute_loss(model, real_output, fake_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, real_output, fake_output, sample, reduce=True):
        tgt_lengths = sample['net_input']['tgt_lengths']

        # print(tgt_lengths[0])
        # print(sample['target'].size())
        # print(sample['masks'].size())
        # print(type(tgt_lengths))
        # print(tgt_lengths.size())
        # print("real_output before")
        # print(real_output.size())
        # print("fake_output before")
        # print(fake_output.size())
        # return None
        start_shape = real_output.size()

        real_output = real_output.view(-1, real_output.size(-1))
        fake_output = real_output.view(-1, fake_output.size(-1))

        output = torch.cat((real_output, fake_output), dim=0).view((-1,))
        # print("output.size ", output.size())
        target = torch.cat((torch.ones(real_output.size(0), dtype=torch.long, device=real_output.device),
                            torch.zeros(fake_output.size(0), dtype=torch.long, device=fake_output.device)), dim=0)
        # print("target size ", target.size())
        loss = F.binary_cross_entropy_with_logits(output, target.float(), reduction='none') #ignore_index=self.padding_idx
        # print("loss.size ", loss.size())
        # print("mask size ", sample['masks'].size())
        new_mask = sample['masks'][:, :start_shape[1]]
        # print("new mask size ", new_mask.size())
        loss = loss.view((start_shape[0], start_shape[1], 2))
        # print("new loss size ", loss.size())
        loss = loss * (new_mask[:, :, None])
        # loss[:, :, 0] = loss[:, :, 0] * (1. - new_mask)
        # loss[:, :, 1] = loss[:, :, 1] * (1. - new_mask)
        # loss = loss * (1. - sample['masks'])
        loss = torch.sum(loss) / torch.sum(new_mask)
        return loss, loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
