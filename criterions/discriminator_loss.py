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
        real_output = real_output.view(-1, real_output.size(-1))
        fake_output = real_output.view(-1, fake_output.size(-1))
        output = torch.cat((real_output, fake_output), dim=0).view((-1,))
        target = torch.cat((torch.ones(real_output.size(0), dtype=torch.long, device=real_output.device),
                            torch.zeros(fake_output.size(0), dtype=torch.long, device=fake_output.device)), dim=0)
        loss = F.binary_cross_entropy_with_logits(output, target.float(), size_average=False, reduce=reduce) #ignore_index=self.padding_idx
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
