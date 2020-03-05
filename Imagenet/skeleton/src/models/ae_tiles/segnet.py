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
    def __init__(self, num_classes ,n_init_features=3,
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
                                          encoder_n_layers[i]))
            self.classifier = SecondClassifier(num_classes = num_classes)

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i]))

            self.final_layer = nn.Conv2d(64,n_init_features,1)

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

        sample['output'] = self.final_layer(feat)
        sample['probs'] = out
        sample['softmax_output'] = F.softmax(out, dim = 1)
        return sample

class SecondClassifier(nn.Module):

    def __init__(self, num_classes, input_chn = 512, output_chn = 128):
        super(SecondClassifier, self).__init__()
        self.layer1 = nn.Conv2d(input_chn, input_chn // 2, 3, 1, 1)
        self.layer2 = nn.Conv2d(input_chn // 2, output_chn, 3, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(9 * output_chn *2*2, num_classes)


    def forward(self, x):
        # shape is B*9,512,3,3
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        # reshape to Bx9*128x2x2
        B,C,H,W = x.shape
        x = x.view(-1, 9*128*H*W)
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    ''' This classifier is supposed to be attached after the encoder
        It has one Conv and two fully connected layers'''

    def __init__(self, num_classes = 1000, input_chn = 9*512, output_chn = 512*3 ):
        super(Classifier, self).__init__()

        # use three convolutions with mix of 1x1 and 3x3 kernels
        self.layer1 = Separable(input_chn, 6*512)       # ~ 35.4 M parameters
        self.layer2 = Separable(6*512, output_chn)      # ~ 14.2 M parameters

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(output_chn*1*1, num_classes)  # ~ 1.5 M parameters ---- total 51.1 M

    def forward(self, x):
        # reshape input tensor to
        B,C,H,W = x.shape 

        # reshape to Bx9*512x3x3
        x = x.reshape((-1,9*C,H,W))

        x = self.layer2(self.layer1(x))
        x = self.avg_pool(x)
        B,C,H,W = x.shape
        x = x.reshape((-1,x.shape[1]*H*W))

        x = self.fc1(x)
        return x


class Separable(nn.Module):
    
    def __init__(self, in_ch, out_ch, kernel = 3):
        super(Separable, self).__init__()
        layers = []
        
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size = kernel, groups = in_ch, padding = 1))
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size = 1))
        layers.append(nn.BatchNorm2d(in_ch))
        layers.append(nn.ReLU(inplace = True))
        layers.append(nn.Conv2d(in_ch, in_ch, kernel_size = kernel, groups = in_ch, padding = 1))
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size = 1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace = True))
        
        self.layer = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layer(x)


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
