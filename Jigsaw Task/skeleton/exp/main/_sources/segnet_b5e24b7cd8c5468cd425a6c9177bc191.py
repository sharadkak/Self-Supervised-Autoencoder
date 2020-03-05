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
    def __init__(self, num_classes ,n_init_features=3, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512), batch_size = 64):
        super(SegNet, self).__init__()
        self.bs = batch_size

        # this mean should be mean for each channel for whole imagenet dataset
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
                                          encoder_n_layers[i], drop_rate))
            self.classifier = Classifier(num_classes = num_classes)

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

    def forward(self, sample):
        indices = []
        unpool_sizes = []

        # change the shape of the input to (32 * 9, 3, 75, 75) and convert it to cuda float tensors
        sample['label'] = sample['label'].reshape((self.bs,))
        feat = sample['image']
        feat = feat.reshape((self.bs * 9, 3, 96, 96))

        # this is image standarization, subtracting each image in bath from mean and divide by std
        feat = feat.type(torch.cuda.FloatTensor).sub(self.mean).div(self.std)
        sample['target_image'] = sample['target_image'].reshape((self.bs * 9, 3, 96, 96))
        sample['target_image'] = sample['target_image'].type(torch.cuda.FloatTensor).sub(self.mean).div(self.std)

        
        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # add a classifier
        out = self.classifier(feat)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        sample['output'] = feat
        sample['probs'] = out
        return sample

class Classifier(nn.Module):
    ''' This classifier is supposed to be attached after the encoder
        It has one Conv and two fully connected layers'''

    def __init__(self, num_classes = 1000, input_chn = 9*512, output_chn = 512*3 ):
        super(Classifier, self).__init__()

        # use three convolutions with mix of 1x1 and 3x3 kernels
        self.layer1 = nn.Sequential(nn.Conv2d(input_chn, 6*512 ,3,1,1),
                        nn.ReLU(inplace = True),
                        nn.BatchNorm2d(6*512),
                        nn.Conv2d( 6*512, output_chn, 3,1,1),
                        nn.ReLU(inplace = True))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(output_chn*1*1, num_classes)

    def forward(self, x):
        # reshape input tensor to
        B,C,H,W = x.shape 
        # input shape is Bx512x3x3
        x = x.reshape((-1,9*C,H,W))

        x = self.layer1(x)
        x = self.avg_pool(x)
        B,C,H,W = x.shape
        x = x.reshape((-1,x.shape[1]*H*W))

        x = self.fc1(x)
        #x = F.softmax(x, dim = 1)
        return x

class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
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
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
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