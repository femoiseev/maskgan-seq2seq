import math
import torch
import torch.nn.functional as F
from copy import deepcopy

from fairseq import utils

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.sequence_generator import SequenceGenerator

from tasks.mask_discriminator_task import MaskDiscriminatorTask


@register_criterion('generator_loss')
class MaskGeneratorCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        self.source_dictionary = task.source_dictionary
        self.target_dictionary = task.target_dictionary
        self.discriminator = self.load_pretrained_discriminator(args.discriminator_path)
        if not args.cpu:
            self.discriminator.cuda()

        self.beam_generator = SequenceGenerator(task.target_dictionary, beam_size=1)
        self.gamma = args.gamma

    @staticmethod
    def add_args(parser):
        super(MaskGeneratorCriterion, MaskGeneratorCriterion).add_args(parser)
        parser.add_argument('--discriminator-path', type=str, help='path to trained discriminator')
        parser.add_argument('--gamma', type=float, default=0.0)

    def load_pretrained_discriminator(self, path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        if not(arg_overrides is None):
            args = utils.override_model_args(args, arg_overrides)
        src_dict = self.source_dictionary
        tgt_dict = self.target_dictionary
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        task = MaskDiscriminatorTask(args, src_dict, tgt_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        logits = model(**sample['net_input'])[0]

        generated = self.beam_generator.generate((model,), sample)
        max_len = sample['target'].shape[1]
        tokens = [x[0]['tokens'] for x in generated]
        lengths = [min(max_len, x.shape[0]) for x in tokens]
        generated_tokens = torch.stack(tuple([torch.cat(
            (
                sample['target'].new_full(
                    (max_len - length,),
                    self.target_dictionary.pad()
                ),
                x[:length],
            )
        ) for x, length in zip(tokens, lengths)]))

        loss, mean_prob = self.compute_loss(model, logits, generated_tokens, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'mean_prob': mean_prob.item(),
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, logits, generated_tokens, sample, reduce=True):
        new_mask = sample['masks']

        fake_input = deepcopy(sample['net_input'])
        fake_input['prev_output_tokens'] = generated_tokens
        discriminator_logits = self.discriminator(**fake_input)[0]
        rewards = F.logsigmoid(discriminator_logits)[:, :, 0]

        logprobs = F.log_softmax(logits, dim=-1)
        generated_shape = generated_tokens.shape
        dict_size  = logprobs.shape[-1]
        num_tokens = generated_shape[0] * generated_shape[1]
        chosen_logprobs = logprobs.reshape(num_tokens, dict_size)[torch.arange(num_tokens),
                                                                  generated_tokens.reshape(-1,)].reshape(generated_shape)
        J = chosen_logprobs * rewards
        J = J * new_mask
        loss = -torch.sum(J) / torch.sum(new_mask)
        probs = torch.exp(rewards) * new_mask
        mean_prob = torch.sum(probs) / torch.sum(new_mask)

        return loss, mean_prob

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        prob_mean = sum(log.get('mean_prob', 0) for log in logging_outputs) / len(logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum,
            'prob_mean': prob_mean,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
