import os.path as pt

# noinspection PyUnresolvedReferences
import simplejpeg
import torch
from torch.backends import cudnn
from torch import nn
from torch import distributed as dist
from datadings.reader import Cycler
from torch.optim import SGD
from math import ceil, floor
from datetime import datetime
from datadings.reader import Cycler
from datetime import datetime
import numpy as np
from crumpets.torch.metrics import MSELossMetric
from crumpets.torch.loss import L1Loss
from crumpets.torch.policy import PolyPolicy
from crumpets.torch.dataloader import TorchTurboDataLoader
from readers import YFCC100mReader
from segnet import SegNet as Net
from worker import FCNWorker
from distributed import distribute
from crumpets.presets import NO_AUGMENTATION, AUGMENTATION_TRAIN
import os , sys
from sacred import Experiment
from sacred.observers import file_storage
from trainer import Trainer
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD


ROOT = pt.abspath(pt.dirname(__file__))
#ROOT = pt.join(ROOT, '..')
EXP_FOLDER = pt.join(ROOT, '../exp')
exp = Experiment(' Experiment: AE with tiles without dropout')


# contruct the loader which will give the batches like a dataloader
# this function should also divide the data for each process
def make_loader(
        file,
        batch_size,
        device,
        world_size,
        rank,
        nworkers,  
        size = 96,
        image_rng=None,
        image_params=None,
        gpu_augmentation=False,
        nsamples = 1000000
):

    # total images are 94502424
    # basically an iterator which cycles over the samples of msgpack
    reader = YFCC100mReader()
    # these many iterations for each dataloader on gpu
    iters = int(floor(len(reader) / world_size))  

    # now reader seeks to the iterator index because dataloader will have images from this index depending on the rank
    reader.seek(iters * rank)
    print('\n start_iteration for device {} is {}'.format(rank ,iters * rank))

    cycler = Cycler(reader)
    
    # crumpets workers expect data to be in msgpack packed dictionaries, this should be done using datalodings library
    worker = FCNWorker(
        ((9*3, 96, 96), np.uint8, IMAGENET_MEAN),
        ((9*3, 96, 96), np.uint8, IMAGENET_MEAN),
        image_params=image_params,
        target_image_params=None,
        image_rng=image_rng,
    )

    return TorchTurboDataLoader(
        cycler.rawiter(), batch_size,
        worker, nworkers,
        gpu_augmentation=gpu_augmentation,
        length= nsamples,
        device=device,
    )


def make_policy(epochs, network, lr, momentum, wd):
    optimizer = SGD([
        {'params': network.parameters(), 'lr': lr},
    ], momentum=momentum, weight_decay=wd)

    # this creates a scheduler which is used to adjust the learning rate, many of the scheduler are defined , choose one as per your need
    # check if the policy is right
    scheduler = PolyPolicy(optimizer, epochs, 1)
    return optimizer, scheduler


# noinspection PyUnusedLocal
@exp.config
def config():
    # number of ranks across all nodes
    world_size = 14
    # first rank on this node
    first_rank = None
    # number of local ranks
    ranks = 14
    # rank of current process
    rank = 0
    # init method, usually TCP address
    init_method = None
    # dataset directory path
    datadir = '/ds2/YFCC100m/image_packs'
    outdir_suffix = 'exp_without_dropout'
    # snapshot directory
    outdir = None
    # batch size per rank
    batch_size = 32
    # number of workers per rank
    num_workers = 4
    # learning rate
    lr = 0.01
    # length of warmup period in epochs
    warmup = 5
    # l2 regularization aka weight decay
    wd = 1e-4
    all_gpu = 1000000
    dataset_size = 94502424
    nsamples = int(floor(all_gpu / world_size))
    num_epochs = int(floor(dataset_size / world_size / nsamples))



log_location = os.path.join(EXP_FOLDER, os.path.basename(sys.argv[0])[:-3])
if len(exp.observers) == 0:
    print('Adding a file observer in %s' % log_location)
    exp.observers.append(file_storage.FileStorageObserver.create(log_location))


@exp.automain
@distribute
def main(
        _run,
        _config,
        world_size,
        rank,
        init_method,
        datadir,
        outdir_suffix,
        batch_size,
        num_workers,
        outdir,
        lr,
        wd,
        num_epochs,
        nsamples
):
    cudnn.benchmark = True
    device = torch.device('cuda:0')  # device is set by CUDA_VISIBLE_DEVICES
    torch.cuda.set_device(device)

    # rank 0 creates experiment observer
    is_master = rank == 0

    # rank joins process group
    print('rank', rank, 'init_method', init_method)
    dist.init_process_group('nccl', rank=rank, world_size=world_size,
                            init_method=init_method)

    # actual training stuff
    train = make_loader(
        pt.join(datadir, '') if datadir else None,
        batch_size,device, world_size, rank, num_workers,
        # this the parameter based on which augmentation is applied to the data
        gpu_augmentation=False, image_rng=None, nsamples = nsamples
    )

    print('\n experiment name ', exp)
    # outdir stuff
    if outdir is None:
        outdir = pt.join(ROOT, '../exp/', outdir_suffix)

    model = Net(batch_size = batch_size)

    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    optimizer, policy = make_policy(num_epochs, model, lr, 0.9, wd)

    # loss for autoencoder
    loss = L1Loss(output_key = 'output' , target_key='target_image').to(device)
    trainer = Trainer(model, optimizer, loss, rank, MSELossMetric(), policy, None, train, None, outdir,
                      snapshot_interval=4 if is_master else None, quiet = True if not is_master else False)


    print('\n Number of epochs are: ', num_epochs)
    start = datetime.now()
    with train:
        trainer.train(num_epochs, start_epoch= 0)

    print("Training complete in: " + str(datetime.now() - start))

