from __future__ import print_function, division, unicode_literals

import logging
import os
import os.path as pt
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from crumpets.presets import AUGMENTATION_TRAIN
from crumpets.logging import JSONLogger
from crumpets.logging import SilentLogger
from crumpets.logging import get_logfilename
from crumpets.logging import make_printer
from crumpets.logging import print
from crumpets.torch.utils import save
from crumpets.torch.metrics import AverageValue
from crumpets.torch.metrics import NoopMetric
from crumpets.torch.policy import NoopPolicy
from crumpets.timing import ETATimer
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD

ROOT = os.path.abspath(os.path.dirname(__file__)) + '/'

def save_image(im, im_type, rank, filename):

    if im_type is 'input':
        dir_name = pt.join(ROOT, '../res/saved_test_input/')
        filename = '{}_im_rank_{}_{}.png'.format(im_type, rank, filename)
        im = im.view(9, 3, 96, 96)[0]
        im = im.cpu().numpy()
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        im.save(dir_name + filename)

    elif im_type is 'output':
        # this is float32
        dir_name = pt.join(ROOT, '../res/saved_test_output/')
        filename = '{}_im_rank_{}_{}.png'.format(im_type, rank, filename)
        im = im.detach().cpu().numpy()
        im = np.transpose(im, (1, 2, 0))
        
        im = (IMAGENET_STD * im) + IMAGENET_MEAN
        im = np.clip(im, 0, 255)
        im = Image.fromarray(im.astype('uint8'))
        im.save( dir_name + filename)


class Trainer(object):
    """
    The Trainer can be used to train a given network.
    It alternately trains one epoch and validates
    the resulting net one epoch.
    Given loss is evaluated each batch,
    gradients are computed and optimizer used to updated weights.
    The loss is also passed to the policy,
    which might update the learning rate.
    Useful information about the training
    flow is regularly printed to the console,
    including an estimated time of arrival.
    Loss, metric and snapshots per epoch are also logged in outdir,
    for later investigation.
    outdir is created if either quiet is `False` or `snapshot_interval > 0`.

    :param network:
        Some network that is to be trained.
        If multiple gpus are used (i.e. multiple devices passed to the data loader)
        a ParallelApply module has to be wrapped around.
    :param optimizer:
        some torch optimzer, e.g. SGD or ADAM, given the network's parameters.
    :param loss:
        some loss function, e.g. CEL or MSE. Make sure to use crumpets.torch.loss
        or implement your own ones, but do not use torch losses directly, since
        they are not capable of handling crumpets sample style (i.e dictionaries).
    :param metric:
        some metric to further measure network's quality.
        Similar to losses, use crumpets.torch.metrics
    :param train_policy:
        some policy to maintain learning rates and such,
        in torch usually called lr_schedulers.
        After each iteration it, given the current loss,
        updates learning rates and potentially other hyperparameters.
    :param val_policy:
        same as train_policy, but updates after validation epoch.
    :param train_iter:
        iterator for receiving training samples,
        usually this means a :class:`~TorchTurboDataLoader` instance.
    :param val_iter:
        same as train_iter, but for retrieving validation samples.
    :param outdir:
        Output directory for logfiles and snapshots.
        Is created including all parent directories if it does not exist.
    :param val_loss:
        same as loss, but applied during validation.
        Default is None, which results in using loss again for validation.
    :param val_metric:
        same as metric, but applied during validation.
        Default is None, which results in using metric again for validation.
    :param snapshot_interval:
        Number of epochs between snapshots.
        Set to 0 or `None` to disable snapshots.
        Default is 1, which means taking a snapshot after every epoch.
    :param quiet:
        If True, trainer will not print to console and will not attempt
        to create a logfile.
    """
    def __init__(
            self,
            network,
            optimizer,
            loss,
            classfier_loss,
            rank,
            metric,
            train_policy,
            val_policy,
            train_iter,
            val_iter,
            outdir,
            val_loss=None,
            val_metric=None,
            snapshot_interval=1,
            quiet=False,
    ):
        self.state = {
            'epoch': 0,
            'network': network,
            'optimizer': optimizer,
            'train_policy': train_policy or NoopPolicy(),
            'val_policy': val_policy or NoopPolicy(),
            'loss': loss,
            'classfier_loss': classfier_loss,
            'rank':rank,
            'metric': metric or NoopMetric(),
            'train_iter': train_iter,
            'val_iter': val_iter,
            'train_metric_values': [],
            'val_metric_values': [],
            'outdir': outdir,
            'val_loss': val_loss,
            'val_metric': val_metric,
            'snapshot_interval': snapshot_interval,
            'quiet': quiet,
        }
        self.hooks = defaultdict(list)
        if outdir is not None and (not quiet or snapshot_interval):
            os.makedirs(outdir, exist_ok=True)
        if not quiet and outdir is not None:
            logpath = pt.join(outdir, get_logfilename('training_'))
            self.logger = JSONLogger('trainer', logpath)
        else:
            self.logger = SilentLogger()

    def add_hook(self, name, fun):
        """
        Add a function hook for the given event.
        Function must accept trainer `state` dictionary as first
        positional argument the current, as well as further keyword
        arguments depending on the type of hook.

        The following events are available during training:

        - `'train_begin'`: run at the beginning of a training epoch
        - `'train_end'`: run after a training epoch has ended
        - `'train_pre_forward'`: run before the forward step;
          receives kwarg `sample`
        - `'train_forward'`: run after the forward step;
          receives kwargs `metric`, `loss`, and `output`
        - `'train_backward'`: run after the backward step;
          receives kwargs `metric`, `loss`, and `output`

        During validation the following hooks are available:

        - `'val_begin'`: run at the beginning of a training epoch
        - `'val_end'`: run after a training epoch has ended
        - `'val_pre_forward'`: run before the forward step;
          receives kwarg `sample`
        - `'val_forward'`: run after the forward step;
          receives kwargs `metric`, `loss`, and `output`

        :param name:
            The event name.
            See above for available hook names and when they are executed.
        :param fun:
            A function that is to be invoked when given event occurs.
            See above for method signature.
        """
        self.hooks[name].append(fun)

    def remove_hook(self, name, fun):
        """
        Remove the function hook with the given name.

        :param name:
            type of hook to remove
        :param fun:
            hook function object to remove
        :return:
        """
        self.hooks[name].remove(fun)

    def _run_hooks(self, name, *args, **kwargs):
        """
        invokes functions hooked to event ``name`` with parameters *args and **kwargs.
        """
        for fun in self.hooks[name]:
            fun(self.state, *args, **kwargs)

    def train(self, num_epochs, start_epoch=0):
        """
        starts the training, logs loss and metrics in logging file and prints progress
        in the console, including an ETA. Also stores snapshots of current model each epoch.

        :param num_epochs:
            number of epochs to train
        :param start_epoch:
            the first epoch, default to 0.
            Can be set higher for finetuning, etc.
        """
        try:
            rem = ETATimer(num_epochs - start_epoch)
            for epoch in range(start_epoch+1, num_epochs+1):
                self.state['epoch'] = epoch
                if not self.state['quiet']:
                    print('Epoch', epoch)
                self.print_info(epoch)
                train_metrics = self.train_epoch()
                self.logger.info(epoch=epoch, phase='train', metrics=train_metrics)
                if self.state['val_iter'] is not None:
                    val_metrics = self.validate_epoch(epoch)
                    self.logger.info(epoch=epoch, phase='val', metrics=val_metrics)
                self.snapshot(epoch)
                if not self.state['quiet']:
                    print('ETA:', rem())
            return self.state
        except Exception as e:
            logging.exception(e)
            raise
        finally:
            # save loss plot
            path = pt.join(ROOT, '../res/plots/')
            losses = [k['total_loss'] for k in self.state['train_metric_values']]
            ae_loss = [k['AE_loss'] for k in self.state['train_metric_values']]
            cl_loss = [k['classifier_loss'] for k in self.state['train_metric_values']]
            print(cl_loss)
            fig = plt.figure()
            plt.plot(losses)
            plt.plot(ae_loss)
            plt.plot(cl_loss)
            plt.legend(('loss', 'AE_loss', 'classifier_loss'))
            plt.savefig(path + 'jigsaw_loss.png')
            plt.close(fig)
            self.logger.info(msg='Finished!')

    def _param_groups(self):
        return self.state['optimizer'].param_groups

    def _lrs(self):
        return [g['lr'] for g in self._param_groups()]

    def print_info(self, epoch):
        """
        prints and logs current learning rates as well as the epoch.

        :param epoch: the current epoch.
        """
        if not self.state['quiet']:
            s = 'learning rates ' + (', '.join(map(str, self._lrs())))
            print(s)
            self.logger.info(epoch=epoch, lrs=self._lrs())

    def snapshot(self, epoch):
        """
        stores snapshot of current model (including optimizer state),
        uses epoch for naming convention (but does always store current model).

        :param epoch: epoch for naming output file
        """
        interval = self.state['snapshot_interval']
        if interval is not None and interval > 0 and epoch % interval == 0:
            path = pt.join(self.state['outdir'], 'epoch_%02d.pth' % epoch)
            save(
                path,
                self.state['train_iter'].iterations,
                self.state['network'],
                self.state['optimizer']
            )

    def train_epoch(self):
        """
        trains one epoch, is invoked by train function. Usually not necessary to be called outside.

        :return: train metric result
        """
        network = self.state['network']
        network = network.train() or network
        optimizer = self.state['optimizer']
        loss = self.state['loss']
        classfier_loss = self.state['classfier_loss']
        loss_metric = AverageValue()
        ae_loss_metric = AverageValue()
        cl_loss_metric = AverageValue()
        metric = self.state['metric']
        metric.reset()
        policy = self.state['train_policy']
        n = self.state['train_iter'].epoch_iterations
        #print('epoch_iterations ',n)
        m = self.state['train_iter'].num_mini_batches

        printer = make_printer(desc='TRAIN', total=n,
                               disable=self.state['quiet'])
        train_metric = dict()
        self._run_hooks('train_begin')
        for iteration, mini_batch in self.state['train_iter']:
            optimizer.zero_grad()
            for sample in mini_batch:
                self._run_hooks('train_pre_forward',
                                sample=sample)
                output = network.forward(sample)
                l = loss(output)

                # at the moment classifier loss is much bigger than autoencoder loss
                l2 = classfier_loss(output)
                # weigh the loss term such that classifier loss does not overwhelm overall loss
                alpha = (l/l2) + 0.15
                total_loss = l + (alpha * l2)
                train_metric.update(
                    metric(output), 
                    AE_loss = ae_loss_metric(l).item(),
                    classifier_loss = cl_loss_metric(l2).item(),
                    total_loss=loss_metric(total_loss).item()      # the AverageMetric gives us the average loss over iterations so far
                )

                if m > 1:
                    total_loss /= m
                self._run_hooks('train_forward',
                                metric=train_metric, loss=total_loss, output=output)
                total_loss.backward()
                self._run_hooks('train_backward',
                                metric=train_metric, loss=total_loss, output=output)
                
                # save two input and output images every epoch
                if iteration % int(n / 2) == 0:

                    save_image(sample['image'][0], 'input', self.state['rank'], str(int(iteration/ self.state['epoch'])))
                    save_image(output['output'][0], 'output', self.state['rank'], str(int(iteration/ self.state['epoch'])))


            # policy depends on iterations and I am not able to control iterations
            policy.step(iteration / n, train_metric['total_loss'])
            #print('Iterations ran ', iteration)
            optimizer.step()
            printer(**train_metric)
        
        self.state['train_metric_values'].append(train_metric)
        self._run_hooks('train_end')
        return train_metric

    def validate_epoch(self, epoch):
        """
        Validate once.
        Invoked by train function.
        Usually not necessary to be called outside.

        :return: val metric result
        """
        network = self.state['network']
        network = network.eval() or network
        loss = self.state['val_loss'] or self.state['loss']
        loss_metric = AverageValue()
        classfier_loss = self.state['classfier_loss']
        metric = self.state['val_metric'] or self.state['metric']
        metric.reset()
        policy = self.state['val_policy']
        n = self.state['val_iter'].epoch_iterations
        printer = make_printer(desc='VAL', total=n,
                               disable=self.state['quiet'])
        val_metric = dict()
        self._run_hooks('val_begin')
        for iteration, mini_batch in self.state['val_iter']:
            for sample in mini_batch:
                self._run_hooks('val_pre_forward',
                                sample=sample)
                with torch.no_grad():
                    output = network.forward(sample)
                    l = loss(output)
                    l2 = classfier_loss(output)
                    l += l2
                val_metric.update(
                    metric(output),
                    loss=loss_metric(l).item(),
                )
                self._run_hooks('val_forward',
                                metric=val_metric, loss=l, output=output)
            printer(**val_metric)
        policy.step(epoch, val_metric['loss'])
        self.state['val_metric_values'].append(val_metric)
        self._run_hooks('val_end')
        return val_metric
