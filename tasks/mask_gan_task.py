from .mask_mle_task import MaskMLETask
from fairseq.tasks import register_task
from fairseq import utils
from fairseq import optim
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

        self.sequence_generator = SequenceGenerator(self.target_dictionary,
                                                    beam_size=1)

        self.discriminator_optimizer = None
        self.discriminator_loss = DiscriminatorCriterion(args, self)
        self.discriminator_steps = args[0].discriminator_steps

    @staticmethod
    def add_args(parser):
        super(MaskGANTask, MaskGANTask).add_args(parser)

        parser.add_argument('--discriminator-steps', type=int, default=3)

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
        if self.discriminator_optimizer is None:
            params = list(filter(lambda p: p.requires_grad,
                                 criterion.discriminator.parameters()))
            self.discriminator_optimizer = optim.build_optimizer(self.args, params)

        discriminator_logging_output = self.train_discriminator(criterion.discriminator, ignore_grad)
        loss, sample_size, generator_logging_output = self.generator_train_step(sample, model, criterion, optimizer)

        logging_output = self.merge_logging_outputs(generator_logging_output, discriminator_logging_output)

        return loss, sample_size, logging_output

    @staticmethod
    def merge_logging_outputs(generator_logging_outputs, discriminator_logging_outputs):
        generator_logging_outputs['discriminator_loss'] = discriminator_logging_outputs['loss']

        return generator_logging_outputs

    def generator_train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
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

        return loss, sample_size, logging_output

    def train_discriminator(self, discriminator, ignore_grad=False):
        logging_output = {}
        for i in range(self.discriminator_steps):
            sample = None

            generated = self.sequence_generator.generate((self.generator,),
                                                         sample)

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

            sample['generated_tokens'] = generated_tokens
            _, _, logging_output = self.discriminator_train_step(discriminator, sample, ignore_grad=ignore_grad)
            self.discriminator_optimizer.step()
            self.discriminator_optimizer.zero_grad()

        return logging_output

    def discriminator_train_step(self, discriminator, sample, ignore_grad=False):
        self.discriminator.train()
        loss, sample_size, logging_output = self.discriminator_loss(discriminator, sample)
        if ignore_grad:
            loss *= 0
        self.discriminator_optimizer.backward(loss)
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