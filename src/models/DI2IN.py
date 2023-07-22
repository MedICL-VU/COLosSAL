import torch
import torch.nn as nn
# from selu import selu


class DI2IN(nn.Module):
    def __init__(self, num_classes=1, nbase=8, norm_layer=nn.BatchNorm3d):
        super(DI2IN, self).__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.conv1_1 = self.encoder(1, 2 * nbase)
        self.conv1_2 = self.encoder(2 * nbase, 2 * nbase, stride=2)

        self.conv2_1 = self.encoder(2 * nbase, 4 * nbase)
        self.conv2_2 = self.encoder(4 * nbase, 4 * nbase, stride=2)

        self.conv3_1 = self.encoder(4 * nbase, 8 * nbase)
        self.conv3_2 = self.encoder(8 * nbase, 8 * nbase, stride=2)

        self.conv4_1 = self.encoder(8 * nbase, 16 * nbase)
        self.conv4_2 = self.encoder(16 * nbase, 16 * nbase, stride=2)

        self.conv5 = self.encoder(16 * nbase, 16 * nbase)
        self.conv6 = self.encoder(16 * nbase, 16 * nbase)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv7_2 = self.encoder(32 * nbase, 8 * nbase)
        self.conv8_2 = self.encoder(16 * nbase, 4 * nbase)
        self.conv9_2 = self.encoder(8 * nbase, 2 * nbase)
        self.conv10_2 = self.encoder(4 * nbase, nbase)

        self.output = self.encoder(nbase, num_classes, kernel_size=1, padding=0, batchnorm=False)
        # self.scale = nn.Sigmoid()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True, bin_selu=False):

        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                self.norm_layer(out_channels),
            )
        elif bin_selu:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.SELU(),
            )
        else:
            layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        return layer

    def forward(self, x, output_feature_only=False):

        syn1 = self.conv1_1(x)
        syn2 = self.conv2_1(self.conv1_2(syn1))
        syn3 = self.conv3_1(self.conv2_2(syn2))
        syn4 = self.conv4_1(self.conv3_2(syn3))
        x = self.conv5(self.conv4_2(syn4))
        x = self.conv6(x)
        x = self.conv7_2(torch.cat([self.upsample(x), syn4], 1))
        x = self.conv8_2(torch.cat([self.upsample(x), syn3], 1))
        x = self.conv9_2(torch.cat([self.upsample(x), syn2], 1))
        x = self.conv10_2(torch.cat([self.upsample(x), syn1], 1))

        return x if output_feature_only else self.output(x)


class DI2IN_BNF(DI2IN):

    def __init__(self, num_classes=1, nbase=8, norm_layer=nn.BatchNorm3d):
        super(DI2IN_BNF, self).__init__(num_classes=num_classes, nbase=nbase, norm_layer=norm_layer)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True, bin_selu=False):

        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                self.norm_layer(out_channels),
                nn.ReLU(),
            )
        elif bin_selu:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.SELU(),
            )
        else:
            layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        return layer


class DI2IN_BNF_ConvTrans(DI2IN_BNF):
    def __init__(self, num_classes=1, nbase=8):
        super(DI2IN_BNF, self).__init__(num_classes=num_classes, nbase=nbase)

        self.upsample_syn4 = nn.ConvTranspose3d(16 * nbase, 16 * nbase, kernel_size=2, stride=2)
        self.upsample_syn3 = nn.ConvTranspose3d(8 * nbase, 8 * nbase, kernel_size=2, stride=2)
        self.upsample_syn2 = nn.ConvTranspose3d(4 * nbase, 4 * nbase, kernel_size=2, stride=2)
        self.upsample_syn1 = nn.ConvTranspose3d(2 * nbase, 2 * nbase, kernel_size=2, stride=2)

        self.conv7_2 = self.encoder(32 * nbase, 8 * nbase)
        self.conv8_2 = self.encoder(16 * nbase, 4 * nbase)
        self.conv9_2 = self.encoder(8 * nbase, 2 * nbase)
        self.conv10_2 = self.encoder(4 * nbase, nbase)

        self.output = self.encoder(nbase, num_classes, kernel_size=1, padding=0, batchnorm=False)

    def forward(self, x):
        syn1 = self.conv1_1(x)
        syn2 = self.conv2_1(self.conv1_2(syn1))
        syn3 = self.conv3_1(self.conv2_2(syn2))
        syn4 = self.conv4_1(self.conv3_2(syn3))

        x = self.conv5(self.conv4_2(syn4))
        x = self.conv6(x)

        x = self.conv7_2(torch.cat([self.upsample_syn4(x), syn4], 1))
        x = self.conv8_2(torch.cat([self.upsample_syn3(x), syn3], 1))
        x = self.conv9_2(torch.cat([self.upsample_syn2(x), syn2], 1))
        x = self.conv10_2(torch.cat([self.upsample_syn1(x), syn1], 1))

        x = self.output(x)
        return x


class DI2IN_DS(nn.Module):
    def __init__(self, num_classes=1, nbase=8, norm_layer=nn.BatchNorm3d):
        super(DI2IN_DS, self).__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.conv1_1 = self.encoder(1, 2 * nbase)
        self.conv1_2 = self.encoder(2 * nbase, 2 * nbase, stride=2)

        self.conv2_1 = self.encoder(2 * nbase, 4 * nbase)
        self.conv2_2 = self.encoder(4 * nbase, 4 * nbase, stride=2)

        self.conv3_1 = self.encoder(4 * nbase, 8 * nbase)
        self.conv3_2 = self.encoder(8 * nbase, 8 * nbase, stride=2)

        self.conv4_1 = self.encoder(8 * nbase, 16 * nbase)
        self.conv4_2 = self.encoder(16 * nbase, 16 * nbase, stride=2)

        self.conv5 = self.encoder(16 * nbase, 16 * nbase)
        self.conv6 = self.encoder(16 * nbase, 16 * nbase)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv7_2 = self.encoder(32 * nbase, 8 * nbase)
        self.conv8_2 = self.encoder(16 * nbase, 4 * nbase)
        self.conv9_2 = self.encoder(8 * nbase, 2 * nbase)
        self.conv10_2 = self.encoder(4 * nbase, nbase)
        self.output = self.encoder(nbase, num_classes, kernel_size=1, padding=0, batchnorm=False)
        
        # deep supervision
        self.aux1 = nn.Conv3d(8 * nbase, num_classes, 1, 1, 0, 1, 1, False)  # 16
        self.aux2 = nn.Conv3d(4 * nbase, num_classes, 1, 1, 0, 1, 1, False)  # 32
        self.aux3 = nn.Conv3d(2 * nbase, num_classes, 1, 1, 0, 1, 1, False)  # 64

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True, bin_selu=False):

        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                self.norm_layer(out_channels),
            )
        elif bin_selu:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.SELU(),
            )
        else:
            layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        return layer

    def forward(self, x):
        syn1 = self.conv1_1(x)
        syn2 = self.conv2_1(self.conv1_2(syn1))
        syn3 = self.conv3_1(self.conv2_2(syn2))
        syn4 = self.conv4_1(self.conv3_2(syn3))

        x = self.conv5(self.conv4_2(syn4))
        x = self.conv6(x)

        x = self.conv7_2(torch.cat([self.upsample(x), syn4], 1))
        aux2 = self.aux1(x)

        x = self.conv8_2(torch.cat([self.upsample(x), syn3], 1))
        aux3 = self.aux2(x)

        x = self.conv9_2(torch.cat([self.upsample(x), syn2], 1))
        aux4 = self.aux3(x)

        x = self.conv10_2(torch.cat([self.upsample(x), syn1], 1))

        if self.training:  # training and use deep supervision
            return [self.output(x), aux4, aux3, aux2]
        else:
            return self.output(x)


        
class DI2IN_KD(nn.Module):
    def __init__(self, num_classes=1, nbase=8, norm_layer=nn.BatchNorm3d):
        super(DI2IN_KD, self).__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.conv1_1 = self.encoder(1, 2 * nbase)
        self.conv1_2 = self.encoder(2 * nbase, 2 * nbase, stride=2)

        self.conv2_1 = self.encoder(2 * nbase, 4 * nbase)
        self.conv2_2 = self.encoder(4 * nbase, 4 * nbase, stride=2)

        self.conv3_1 = self.encoder(4 * nbase, 8 * nbase)
        self.conv3_2 = self.encoder(8 * nbase, 8 * nbase, stride=2)

        self.conv4_1 = self.encoder(8 * nbase, 16 * nbase)
        self.conv4_2 = self.encoder(16 * nbase, 16 * nbase, stride=2)

        self.conv5 = self.encoder(16 * nbase, 16 * nbase)
        self.conv6 = self.encoder(16 * nbase, 16 * nbase)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv7_2 = self.encoder(32 * nbase, 8 * nbase)
        self.conv8_2 = self.encoder(16 * nbase, 4 * nbase)
        self.conv9_2 = self.encoder(8 * nbase, 2 * nbase)
        self.conv10_2 = self.encoder(4 * nbase, nbase)

        self.output = self.encoder(nbase, num_classes, kernel_size=1, padding=0, batchnorm=False)
        # self.scale = nn.Sigmoid()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True, bin_selu=False):

        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                self.norm_layer(out_channels),
            )
        elif bin_selu:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.SELU(),
            )
        else:
            layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        return layer

    def forward(self, x, output_feature=False):

        syn1 = self.conv1_1(x)
        syn2 = self.conv2_1(self.conv1_2(syn1))
        syn3 = self.conv3_1(self.conv2_2(syn2))
        syn4 = self.conv4_1(self.conv3_2(syn3))

        x = self.conv5(self.conv4_2(syn4))
        x = self.conv6(x)

        f1 = self.conv7_2(torch.cat([self.upsample(x), syn4], 1))
        f2 = self.conv8_2(torch.cat([self.upsample(f1), syn3], 1))
        f3 = self.conv9_2(torch.cat([self.upsample(f2), syn2], 1))
        f4 = self.conv10_2(torch.cat([self.upsample(f3), syn1], 1))
    
        if output_feature:
            return f3, self.output(f4)
        else:
            return self.output(f4)


class DI2IN_BNF_ConvTrans_KD(DI2IN_BNF):
    def __init__(self, num_classes=1, nbase=8):
        super(DI2IN_BNF, self).__init__(num_classes=num_classes, nbase=nbase)

        self.upsample_syn4 = nn.ConvTranspose3d(16 * nbase, 16 * nbase, kernel_size=2, stride=2)
        self.upsample_syn3 = nn.ConvTranspose3d(8 * nbase, 8 * nbase, kernel_size=2, stride=2)
        self.upsample_syn2 = nn.ConvTranspose3d(4 * nbase, 4 * nbase, kernel_size=2, stride=2)
        self.upsample_syn1 = nn.ConvTranspose3d(2 * nbase, 2 * nbase, kernel_size=2, stride=2)

        self.conv7_2 = self.encoder(32 * nbase, 8 * nbase)
        self.conv8_2 = self.encoder(16 * nbase, 4 * nbase)
        self.conv9_2 = self.encoder(8 * nbase, 2 * nbase)
        self.conv10_2 = self.encoder(4 * nbase, nbase)

        self.output = self.encoder(nbase, num_classes, kernel_size=1, padding=0, batchnorm=False)

    def forward(self, x, output_feature=False):
        syn1 = self.conv1_1(x)
        syn2 = self.conv2_1(self.conv1_2(syn1))
        syn3 = self.conv3_1(self.conv2_2(syn2))
        syn4 = self.conv4_1(self.conv3_2(syn3))

        x = self.conv5(self.conv4_2(syn4))
        x = self.conv6(x)

        f1 = self.conv7_2(torch.cat([self.upsample_syn4(x), syn4], 1))
        f2 = self.conv8_2(torch.cat([self.upsample_syn3(f1), syn3], 1))
        f3 = self.conv9_2(torch.cat([self.upsample_syn2(f2), syn2], 1))
        f4 = self.conv10_2(torch.cat([self.upsample_syn1(f3), syn1], 1))

        if output_feature:
            return f1, f2, f3, self.output(f4)
        else:
            return self.output(f4)


class DI2IN_BNF_ConvTrans_KD_V2(DI2IN_BNF):
    def __init__(self, num_classes=1, nbase=8):
        super(DI2IN_BNF, self).__init__(num_classes=num_classes, nbase=nbase)

        self.upsample_syn4 = nn.ConvTranspose3d(16 * nbase, 16 * nbase, kernel_size=2, stride=2)
        self.upsample_syn3 = nn.ConvTranspose3d(8 * nbase, 8 * nbase, kernel_size=2, stride=2)
        self.upsample_syn2 = nn.ConvTranspose3d(4 * nbase, 4 * nbase, kernel_size=2, stride=2)
        self.upsample_syn1 = nn.ConvTranspose3d(2 * nbase, 2 * nbase, kernel_size=2, stride=2)

        self.conv7_2 = self.encoder(32 * nbase, 8 * nbase)
        self.conv8_2 = self.encoder(16 * nbase, 4 * nbase)
        self.conv9_2 = self.encoder(8 * nbase, 2 * nbase)
        self.conv10_2 = self.encoder(4 * nbase, nbase)

        self.output = self.encoder(nbase, num_classes, kernel_size=1, padding=0, batchnorm=False)

        self.map_f1 = nn.Conv3d(64, 256, kernel_size=1)
        self.map_f2 = nn.Conv3d(32, 128, kernel_size=1)
        self.map_f3 = nn.Conv3d(16, 64, kernel_size=1)

    def forward(self, x, output_feature=False):
        syn1 = self.conv1_1(x)
        syn2 = self.conv2_1(self.conv1_2(syn1))
        syn3 = self.conv3_1(self.conv2_2(syn2))
        syn4 = self.conv4_1(self.conv3_2(syn3))

        x = self.conv5(self.conv4_2(syn4))
        x = self.conv6(x)

        f1 = self.conv7_2(torch.cat([self.upsample_syn4(x), syn4], 1))
        f2 = self.conv8_2(torch.cat([self.upsample_syn3(f1), syn3], 1))
        f3 = self.conv9_2(torch.cat([self.upsample_syn2(f2), syn2], 1))
        f4 = self.conv10_2(torch.cat([self.upsample_syn1(f3), syn1], 1))

        if output_feature:
            return self.map_f1(f1), self.map_f2(f2), self.map_f3(f3), self.output(f4)
        else:
            return self.output(f4)



class DI2IN_L(nn.Module):
    def __init__(self, num_classes=1, nbase=8, norm_layer=nn.BatchNorm3d):
        super(DI2IN_L, self).__init__()
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        self.conv1_1 = self.encoder(1, 2 * nbase)
        self.conv1_2 = self.encoder(2 * nbase, 2 * nbase, stride=2)

        self.conv2_1 = self.encoder(2 * nbase, 4 * nbase)
        self.conv2_2 = self.encoder(4 * nbase, 4 * nbase, stride=2)

        self.conv3_1 = self.encoder(4 * nbase, 8 * nbase)
        self.conv3_2 = self.encoder(8 * nbase, 8 * nbase, stride=2)

        self.conv4_1 = self.encoder(8 * nbase, 16 * nbase)
        self.conv4_2 = self.encoder(16 * nbase, 16 * nbase, stride=2)

        self.conv5 = self.encoder(16 * nbase, 16 * nbase)
        self.conv6 = self.encoder(16 * nbase, 16 * nbase)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.conv7_2 = self.encoder(32 * nbase, 8 * nbase)
        self.conv8_2 = self.encoder(16 * nbase, 4 * nbase)
        self.conv9_2 = self.encoder(8 * nbase, 2 * nbase)
        self.conv10_2 = self.encoder(4 * nbase, 8 * nbase)  # was nbase

        self.output = self.encoder(8 * nbase, num_classes, kernel_size=1, padding=0, batchnorm=False)
        # self.scale = nn.Sigmoid()

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True, bin_selu=False):

        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                self.norm_layer(out_channels),
            )
        elif bin_selu:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.SELU(),
            )
        else:
            layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

        return layer

    def forward(self, x, output_feature_only=False):

        syn1 = self.conv1_1(x)
        syn2 = self.conv2_1(self.conv1_2(syn1))
        syn3 = self.conv3_1(self.conv2_2(syn2))
        syn4 = self.conv4_1(self.conv3_2(syn3))

        x = self.conv5(self.conv4_2(syn4))
        x = self.conv6(x)

        x = self.conv7_2(torch.cat([self.upsample(x), syn4], 1))
        x = self.conv8_2(torch.cat([self.upsample(x), syn3], 1))
        x = self.conv9_2(torch.cat([self.upsample(x), syn2], 1))
        x = self.conv10_2(torch.cat([self.upsample(x), syn1], 1))

        return x if output_feature_only else self.output(x)
