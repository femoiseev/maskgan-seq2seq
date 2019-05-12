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
        real_input = sample['net_input']
        real_input['prev_output_tokens'] = sample['target']
        real_output = model(**real_input)[0]
        fake_input = deepcopy(sample['net_input'])
        fake_input['prev_output_tokens'] = sample['generated_tokens']
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
        new_mask = sample['masks']
        real_output = real_output.squeeze()
        fake_output = fake_output.squeeze()

        output = torch.cat((real_output, fake_output), dim=0)
        target = torch.cat((torch.ones(real_output.size(), dtype=torch.long, device=real_output.device) - 0.1,
                            torch.zeros(fake_output.size(), dtype=torch.long, device=fake_output.device) + 0.1), dim=0)
        loss = F.binary_cross_entropy_with_logits(output, target.float(), reduction='none')

        num_samples, seq_len = real_output.shape
        loss = loss.view((2, num_samples, seq_len))
        loss = loss * (new_mask[None, :, :])
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
            'loss': loss_sum,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / math.log(2)
        return agg_output
