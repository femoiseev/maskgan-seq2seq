from fairseq import utils
from fairseq.tasks import register_task
from fairseq.sequence_generator import SequenceGenerator
import torch

from tasks.mask_mle_task import MaskMLETask


@register_task("mask_discriminator")
class MaskDiscriminatorTask(MaskMLETask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = self.load_pretrained_generator(args[0].generator_path)
        if not args[0].cpu:
            self.generator.cuda()

        self.__passed_iters = 0
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

    def process_sample(self, sample, p):
        p = 0.5
        mask = torch.distributions.Bernoulli(torch.Tensor([p]))
        target = sample['target'].clone()

        mask_tensor = mask.sample(target.size())[:, :, 0].to("cuda")

        pad_idx = self.target_dictionary.pad()
        mask_idx = self.target_dictionary.index("<MASK>")

        target[(target != pad_idx) & (
            mask_tensor.byte())] = mask_idx
        mask_tensor[(target == pad_idx)] = 0

        sample['net_input']['masked_tgt'] = target
        sample['masks'] = mask_tensor
        return sample

    def __get_mask_rate__(self):
        return torch.clamp(0 + self.__passed_iters * 0.0001, 0., 1.)

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

        p = self.__get_mask_rate__()
        sample = self.process_sample(sample, p=p)

        self.generator.eval()
        model.train()

        generated = self.sequence_generator.generate((self.generator, ), sample)

        max_len = sample['target'].shape[1]
        tokens = [x[0]['tokens'] for x in generated]
        lengths = [min(max_len, x.shape[0]) for x in tokens]
        generated_tokens = torch.stack([torch.cat(
            (
                sample['target'].new_full(
                    (max_len - length,),
                    self.target_dictionary.pad()
                ),
                x[:length],
            )
        ) for x, length in zip(tokens, lengths)])

        sample['generated_tokens'] = generated_tokens
        # return
        # sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens']
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        self.__passed_iters += 1
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        p = self.__get_mask_rate__()
        sample = self.process_sample(sample, p=p)
        self.generator.eval()
        model.eval()
        with torch.no_grad():
            generated = self.sequence_generator.generate((self.generator,), sample)
            max_len = sample['target'].shape[1]
            tokens = [x[0]['tokens'] for x in generated]
            lengths = [min(max_len, x.shape[0]) for x in tokens]
            generated_tokens = torch.stack([torch.cat(
                (
                    sample['target'].new_full((max_len - length ,), self.target_dictionary.pad()),
                    x[:length]
                )
            ) for x, length in zip(tokens, lengths)])
            sample['generated_tokens'] = generated_tokens

            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output


