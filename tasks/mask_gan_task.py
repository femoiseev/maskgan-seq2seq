from .mask_mle_task import MaskMLETask
from fairseq.tasks import register_task
from fairseq import utils
from fairseq.optim import FP16Optimizer
import torch

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.sequence_generator import SequenceGenerator

from tasks.mask_discriminator_task import MaskDiscriminatorTask
from criterions.discriminator_loss import  DiscriminatorCriterion
from copy import deepcopy


@register_task("mask_gan")
class MaskGANTask(MaskMLETask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.discriminator = self.load_pretrained_discriminator(args[0].discriminator_path)
        if not args[0].cpu:
            self.discriminator.cuda()

        self.sequence_generator = SequenceGenerator(self.target_dictionary,
                                                    beam_size=5)

        params = list(filter(lambda p: p.requires_grad,
                             self.discriminator.parameters()))
        self.opt_discr = torch.optim.Adam(params,)
        self.discriminator_loss = DiscriminatorCriterion(args, MaskMLETask)

    # @staticmethod
    # def add_args(parser):
    #     MaskMLETask.add_args(parser)
    #
    #     parser.add_argument('--discriminator-path', type=str, help='path to discriminator')

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

    def train_step(self, sample, model, criterion, optimizer,
                   ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """

        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        discr_step = 3
        for i in range(discr_step):

            sample_for_discriminator = deepcopy(sample)

            generated = self.sequence_generator.generate((self.generator,),
                                                         sample_for_discriminator)

            max_len = sample_for_discriminator['target'].shape[1]
            tokens = [x[0]['tokens'] for x in generated]
            lengths = [min(max_len, x.shape[0]) for x in tokens]
            generated_sents = torch.stack([torch.cat(
                (
                    sample_for_discriminator['target'].new_full(
                        (max_len - length,),
                        self.target_dictionary.pad()
                    ),
                    x[:length],
                )
            ) for x, length in zip(tokens, lengths)])

            sample_for_discriminator['generated_sents'] = generated_sents
            _ = self.train_discr(sample_for_discriminator, ignore_grad=ignore_grad)

        return loss, sample_size, logging_output

    def train_discr(self, sample, ignore_grad=False):
        loss, sample_size, logging_output = self.discriminator_loss(self.discriminator, sample)
        if ignore_grad:
            loss *= 0
        self.opt_discr.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(models, sample,
                                      prefix_tokens=prefix_tokens)