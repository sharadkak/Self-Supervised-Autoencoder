import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from crumpets.presets import IMAGENET_MEAN, IMAGENET_STD


class SegNet(nn.Module):
    """
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """
    def __init__(self, batch_size, n_init_features=3,
                 filter_config=(64, 128, 256, 512, 512)):
        super(SegNet, self).__init__()

        self.bs = batch_size
        self.register_buffer(
            'mean', torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor(IMAGENET_STD).view(1, -1, 1, 1)
        )


        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 2)
        decoder_filter_config = filter_config[::-1] + (64,)
        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i]))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i]))

            self.final_layer = nn.Conv2d(64,n_init_features,1)

    def forward(self, sample):
        indices = []
        unpool_sizes = []

        # change the shape of the input to (32 * 9, 3, 96, 96) and converting tensor to cudafloat tensors
        feat = sample['image']
        feat = feat.reshape((self.bs * 9, 3, 96, 96))
        feat = feat.type(torch.cuda.FloatTensor).sub(self.mean).div(self.std)
        sample['target_image'] = sample['target_image'].reshape((self.bs * 9, 3, 96, 96))
        sample['target_image'] = sample['target_image'].type(torch.cuda.FloatTensor).sub(self.mean).div(self.std)
        
        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        sample['output'] = self.final_layer(feat)
        return sample


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        channels = [n_in_feat] + [n_out_feat for i in range(n_blocks)]
        layers = []
        for i in range(n_blocks):
            layers += [nn.Conv2d(channels[i], channels[i+1], 3, 1, 1),
                  nn.BatchNorm2d(channels[i+1]),
                  nn.ReLU(inplace=True)]



        self.features = nn.Sequential(*layers)


    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2):
        super(_Decoder, self).__init__()

        channels = [n_in_feat for i in range(n_blocks)] + [n_out_feat]
        layers = []
        
        for i in range(n_blocks):
            layers += [nn.Conv2d(channels[i], channels[i+1], 3, 1, 1),
                  nn.BatchNorm2d(channels[i+1]),
                  nn.ReLU(inplace=True)]



        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)
