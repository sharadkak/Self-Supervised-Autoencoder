import simplejpeg
import torch
import torch.nn as nn
import os.path as pt
import sys
import os
from math import floor
import numpy as np
from datetime import datetime
from datadings.reader import Cycler
from datadings.reader import MsgpackReader
from torch.backends import cudnn
from torch.optim import SGD
from distributed import distribute
from torch import distributed as dist
import torchvision.models as models
from crumpets.torch.loss import CrossEntropyLoss
from crumpets.torch.metrics import AccuracyMetric
from trainer import Trainer
from crumpets.torch.policy import PolyPolicy
from crumpets.torch.dataloader import TorchTurboDataLoader
from crumpets.workers import ClassificationWorker
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD
from crumpets.presets import AUGMENTATION_TRAIN
from crumpets.torch.utils import Unpacker
from sacred import Experiment
from sacred.observers import file_storage
from models.Jigsaw.segnet import SegNet as Net2
from models.vanilla.segnet import SegNet as Net1
from models.ae_tiles.segnet import SegNet as Net3

class Normalize(nn.Module):
    # noinspection PyUnresolvedReferences
    def __init__(self, module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        nn.Module.__init__(self)
        self.module = module
        self.register_buffer(
            'mean', torch.FloatTensor(mean).view(1, len(mean), 1, 1)
        )
        self.register_buffer(
            'std', torch.FloatTensor(std).view(1, len(std), 1, 1)
        )

    def forward(self, x):
        x = x.float()  # implicitly convert to float
        x = x.sub(self.mean).div(self.std)
        return self.module(x)


ROOT = pt.abspath(pt.dirname(__file__))

exp = Experiment('Experiment: Jigsaw encoder on Imagenet')
# Add a FileObserver if one hasn't been attached already
EXP_FOLDER = '../exp/'
log_location = pt.join(EXP_FOLDER, pt.basename(sys.argv[0])[:-3])
if len(exp.observers) == 0:
    print('Adding a file observer in %s' % log_location)
    exp.observers.append(file_storage.FileStorageObserver.create(log_location))


def get_checkpoint(exp, exp_variant):
	# currently three experiments exist, vanilla_AE, AE_with_tiles, Jigsaw (two variants)
	# have namedtuple for experiment and checkpoints directory

	if exp is None:
		raise RuntimeError('Experiment can not be None!')

	_dir = '/netscratch/kakran/'
	suffix = pt.join(exp, 'skeleton/exp/')

	file_dirs = {'static_weigh': 'jigsaw_static_weight',
	'mtl_loss': 'jigsaw_mtl_loss',
	'per_tile_aug': 'per_tile_aug',
	'vanilla_ae': 'logs_and_snapshot',
	'ae_with_tiles': 'exp_without_dropout'}

	path = pt.join(_dir, suffix, file_dirs[exp_variant])
	print('exp path is ', path)
	checkpoint = torch.load(pt.join(path, 'epoch_90.pth'))
	if exp == 'Jigsaw':
		model = Net2(num_classes = 500, batch_size = 64)
	elif exp == 'AE_with_tiles':
		model = Net3(batch_size = 64)
	elif exp == 'Vanilla_Autoenoder':
		model = Net1()
	else:
		print('specify a proper experiment!')

	return model, checkpoint

def get_network(exp = None, exp_variant = None):
	# call checkpoint and construct the VGG from that
	# load vgg from the pytorch and just assign it weights from your checkpoint
	if exp == 'baseline':
		vgg = models.vgg16_bn(pretrained = False)
		return vgg, None, None

	vgg = models.vgg16_bn(pretrained = False)
	model, checkpoint = get_checkpoint(exp, exp_variant)

	return vgg, model, checkpoint 


def make_loader(
        file,
        batch_size,
        world_size,
        rank,
        num_mini_batches = 1,
        nworkers = 4,
        image_rng= None,
        image_params=None,
        gpu_augmentation=True,
        device = 'cuda:0'
):
	
    reader = MsgpackReader(file)
    iters = int(floor(len(reader) / world_size ))
    reader.seek(iters * rank)
    print('\n nsamples for each gpu: ', iters)
    print('\n device {} starting index is: {} '.format(rank, iters * rank))
    cycler = Cycler(reader)
    worker = ClassificationWorker(
        ((3, 224, 224), np.uint8, IMAGENET_MEAN),
        ((1,), np.int),
        image_params=image_params,
        image_rng=image_rng,
    )
    return TorchTurboDataLoader(
        cycler.rawiter(), batch_size,
        worker, nworkers,
        gpu_augmentation=gpu_augmentation,
        length=iters,
        device= device,
    )


def make_policy(epochs, network, lr, momentum, wd):
    optimizer = SGD(filter(lambda p: p.requires_grad, network.parameters()), lr = lr, momentum=momentum, weight_decay=wd)
    scheduler = PolyPolicy(optimizer, epochs, 1)
    return optimizer, scheduler

@exp.config
def config():
	'''experiment parameter could take any of these values.  Vanilla_Autoenoder, AE_with_tiles, Jigsaw'''
	experiment = 'Jigsaw'
	experiment_variant = 'static_weigh'
	world_size = 1
	ranks = 1
	first_rank = None
	rank = 0 
	init_method = None
	data_dir = '/netscratch/folz/ILSVRC12_opt/'
	batch_size = 64
	val_batch_size = 64
	out_dir = pt.join(ROOT, '../exp/logs_and_snapshot')
	out_suffix = None
	momentum = 0.9
	wd = 1e-4
	lr = 0.01
	epochs = 3
	device = 'cuda:0'

@exp.automain
@distribute
def main(_run,
	_config,
	experiment,
	experiment_variant,
	world_size,
	rank,
	init_method,
	data_dir,
	batch_size,
	val_batch_size,
	out_dir,
	momentum,
	wd,
	lr,
	epochs,
	device):

	cudnn.benchmark = True
	torch.cuda.set_device(device)
	is_master = rank == 0
	dist.init_process_group('nccl', rank = rank, world_size = world_size,
		init_method = init_method)


	out_dir = pt.join(out_dir, experiment_variant)
	if not pt.exists(out_dir):
		os.makedirs(out_dir)

	train = make_loader(pt.join(data_dir , 'train.msgpack'), batch_size, world_size,
	 rank, image_rng=AUGMENTATION_TRAIN)
	val = make_loader(pt.join(data_dir , 'val.msgpack'), val_batch_size, world_size,
	 rank, image_params={'scale': 256/224})

	vgg, network,checkpoint = get_network(experiment, experiment_variant)

	# if pretrained networ is returned, load it's encoder parameters in vgg16
	if network not None:

		network = network.to(device)
		network = nn.parallel.DistributedDataParallel(network, device_ids=[device])
		network.load_state_dict(checkpoint['model_state'])
		vgg_state_dict = vgg.state_dict()
		net_state_dict = network.state_dict()

		_ = net_state_dict.pop('module.mean')
		_ = net_state_dict.pop('module.std')

		for (k1,v1),(k2,v2) in zip(vgg_state_dict.items(), net_state_dict.items()):
			if 'encoders' in k2 and (v1.shape == v2.shape):
				#copy the tensors
				v1.data.copy_(v2.data)
			else:
				break

		# now load this updated state_dict into the vgg
		vgg.load_state_dict(vgg_state_dict)

		# only freeze the conv layers which is the first child in network
		ct = 0
		for child in vgg.children():
			if ct < 1:
				for name, param in child.named_parameters():
					param.requires_grad = False

			ct += 1

	# wrap the vgg network with Normalize and Unpacker
	vgg = Normalize(module = vgg).to(device)
	vgg = vgg.to(device)
	vgg = nn.parallel.DistributedDataParallel(vgg, device_ids=[device])
	vgg = Unpacker(vgg)

	# define optimizer with parameters which only require gradients
	optimizer, policy = make_policy(epochs, vgg, lr, momentum, wd)

	loss = CrossEntropyLoss(target_key = 'label').to(device)

	trainer = Trainer(vgg, optimizer, loss, AccuracyMetric(output_key = 'probs'), policy, None, train, val, 
		out_dir, snapshot_interval=5 if is_master else None, quiet = True if not is_master else False)

	start = datetime.now()
	with train:
		with val:
			trainer.train(epochs, start_epoch = 0)

	print('Total Time taken: ', datetime.now() - start)

