from fairseq import utils, options
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks import register_task
from dataloaders.mle_dataset import MLELanguagePairDataset
from fairseq.data import (IndexedCachedDataset,
                          IndexedDataset,
                          IndexedRawTextDataset,
                          LanguagePairDataset,
                          ConcatDataset,
                          data_utils)
from fairseq.sequence_generator import SequenceGenerator
import os
import torch
import itertools

from tasks.mask_mle_task import MaskMLETask
from models import MaskTransformer


@register_task("mask_discriminator")
class MaskDiscriminatorTask(MaskMLETask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = self.load_pretrained_generator(args[0].generator_path)

        if not args[0].cpu:
            self.generator.cuda()

        self.sequence_generator = SequenceGenerator(self.target_dictionary, beam_size=5)

    @staticmethod
    def add_args(parser):
        MaskMLETask.add_args(parser)

        parser.add_argument('--generator-path', type=str, help='path to trained generator')

    def load_pretrained_generator(self, path, arg_overrides=None):
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

        task = MaskMLETask(args, src_dict, tgt_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
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
        self.generator.eval()
        model.train()

        generated = self.sequence_generator.generate((self.generator, ), sample)
        max_len = sample['target'].shape[1] - 1
        tokens = [x[0]['tokens'] for x in generated]
        lengths = [min(max_len, x.shape[0]) for x in tokens]
        generated_sents = torch.stack([torch.cat((x[:length], sample['target'].new_full((max_len - length,), self.target_dictionary.pad()))) for x, length in zip(tokens, lengths)])
        sample['generated_sents'] = generated_sents

        sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'][:, 1:]
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        self.generator.eval()
        model.eval()
        with torch.no_grad():
            generated = self.sequence_generator.generate((self.generator,), sample)
            max_len = sample['target'].shape[1] - 1
            tokens = [x[0]['tokens'] for x in generated]
            lengths = [min(max_len, x.shape[0]) for x in tokens]
            generated_sents = torch.stack([torch.cat((x[:length], sample['target'].new_full((max_len - length,), self.target_dictionary.pad()))) for x, length in zip(tokens, lengths)])
            sample['generated_sents'] = generated_sents

            sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'][:, 1:]
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

