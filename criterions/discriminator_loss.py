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
        # return None

        print("#" * 40)
        print("compute loss block")

        print("real_output before")
        print(real_output.size())
        print("fake_output before")
        print(fake_output.size())

        print('real output[0]')
        print(real_output[0])
        print('fake output')
        print(fake_output[0])
        print("masks")
        print(sample['masks'][0])
        print('masks size')
        print(sample['masks'].size())
        print("#" * 40)

        start_shape = real_output.size()
        new_mask = sample['masks'][:, :start_shape[1]]
        print('new masks size')
        print(new_mask.size())
        print("median real on mask 1")
        print(torch.median(real_output * new_mask.unsqueeze(-1)))
        print("median fake on mask 1")
        print(torch.median(fake_output * new_mask.unsqueeze(-1)))

        start_shape = real_output.size()

        real_output = real_output.view(-1, real_output.size(-1))
        fake_output = real_output.view(-1, fake_output.size(-1))

        output = torch.cat((real_output, fake_output), dim=0).view((-1,))
        # print("output.size ", output.size())
        target = torch.cat((torch.ones(real_output.size(0), dtype=torch.long, device=real_output.device),
                            torch.zeros(fake_output.size(0), dtype=torch.long, device=fake_output.device)), dim=0)
        # print("target size ", target.size())
        loss = F.binary_cross_entropy_with_logits(output, target.float(), reduction='none') #ignore_index=self.padding_idx

        # output =  torch.cat((0.99999 * torch.ones(real_output.size(0), dtype=torch.float, device=real_output.device),
        #                     0.000001 + torch.zeros(fake_output.size(0), dtype=torch.float, device=fake_output.device)), dim=0)

        # output = torch.log(output)

        output = 0.001 + torch.distributions.Bernoulli(0.5 * torch.ones(output.size(0))).sample().cuda() * 0.99

        print("output size ", output.size())
        print("target size ", target.size())
        stupid_loss = F.binary_cross_entropy(output, target.float(), reduction='none')
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

        stupid_loss = stupid_loss.view((start_shape[0], start_shape[1], 2))
        stupid_loss = stupid_loss * (new_mask[:, :, None])
        stupid_loss = torch.sum(stupid_loss) / torch.sum(new_mask)
        print("stupid loss {:.5f}".format(stupid_loss.item()))
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
