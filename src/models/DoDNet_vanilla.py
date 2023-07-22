#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 07/26/2022


import torch.nn as nn
from torch.nn import functional as F
import torch

in_place = True


# DoDNet implementation adapted from https://github.com/jianpengz/DoDNet/blob/main/a_DynConv/unet3D_DynConv882.py


class Conv3d(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=1, dilation=1, bias=False, weight_std=False):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1, weight_std=False):
        super(NoBottleneck, self).__init__()
        self.weight_std = weight_std
        self.gn1 = nn.GroupNorm(16, inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.relu = nn.ReLU(inplace=in_place)

        self.gn2 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1, padding=(1,1,1),
                                dilation=dilation * multi_grid, bias=False, weight_std=self.weight_std)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.gn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.gn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class unet3D(nn.Module):
    def __init__(self, layers, num_classes=3, weight_std = False):
        self.inplanes = 128
        self.weight_std = weight_std
        super(unet3D, self).__init__()

        self.conv1 = conv3x3x3(1, 32, stride=[1, 1, 1], weight_std=self.weight_std)

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, layers[0], stride=(1, 1, 1))
        self.layer1 = self._make_layer(NoBottleneck, 32, 64, layers[1], stride=(2, 2, 2))
        self.layer2 = self._make_layer(NoBottleneck, 64, 128, layers[2], stride=(2, 2, 2))
        self.layer3 = self._make_layer(NoBottleneck, 128, 256, layers[3], stride=(2, 2, 2))
        self.layer4 = self._make_layer(NoBottleneck, 256, 256, layers[4], stride=(2, 2, 2))

        self.fusionConv = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            conv3x3x3(256, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0), weight_std=self.weight_std)
        )

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.x8_resb = self._make_layer(NoBottleneck, 256, 128, 1, stride=(1, 1, 1))
        self.x4_resb = self._make_layer(NoBottleneck, 128, 64, 1, stride=(1, 1, 1))
        self.x2_resb = self._make_layer(NoBottleneck, 64, 32, 1, stride=(1, 1, 1))
        self.x1_resb = self._make_layer(NoBottleneck, 32, 32, 1, stride=(1, 1, 1))

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 32),
            nn.ReLU(inplace=in_place),
            nn.Conv3d(32, 8, kernel_size=1)
        )

        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool3d((1,1,1))
        )
        # self.controller = nn.Conv3d(256+7, 162, kernel_size=1, stride=1, padding=0)  # pelvic
        # self.controller = nn.Conv3d(256+12, 162, kernel_size=1, stride=1, padding=0)  # 8x8, 8x8, 8x1, 1x8, 1x8, 1x1  ja_cm
        self.controller = nn.Conv3d(256+6, 162, kernel_size=1, stride=1, padding=0)  # 8x8, 8x8, 8x1, 1x8, 1x8, 1x1  ja_cm
        # self.controller = nn.Conv3d(256+2, 162, kernel_size=1, stride=1, padding=0)  # 8x8, 8x8, 8x1, 1x8, 1x8, 1x1  ja_cm

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1), dilation=1, multi_grid=1):
        downsample = None
        if stride[0] != 1 or stride[1] != 1 or stride[2] != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.GroupNorm(16, inplanes),
                nn.ReLU(inplace=in_place),
                conv3x3x3(inplanes, planes, kernel_size=(1, 1, 1), stride=stride, padding=0,
                          weight_std=self.weight_std),
            )

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid), weight_std=self.weight_std))
        # self.inplanes = planes
        for i in range(1, blocks):
            layers.append(
                block(planes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid),
                      weight_std=self.weight_std))

        return nn.Sequential(*layers)

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        # task_encoding = torch.zeros(size=(N, 12))
        # task_encoding = torch.zeros(size=(N, 7))
        task_encoding = torch.zeros(size=(N, 6))
        # task_encoding = torch.zeros(size=(N, 2))
        for i in range(N):
            task_encoding[i, task_id[i]]=1                                                 
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)  # 1 is the output channel

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    # def forward(self, input, task_id):
    #     # clipping the batch size of task_id for SlidingWindowInference
    #     task_id = task_id[:input.size()[0]]
    #     # print(input.size(), task_id.size())

    #     x = self.conv1(input)
    #     x = self.layer0(x)
    #     skip0 = x

    #     x = self.layer1(x)
    #     skip1 = x

    #     x = self.layer2(x)
    #     skip2 = x

    #     x = self.layer3(x)
    #     skip3 = x

    #     x = self.layer4(x)

    #     x = self.fusionConv(x)

    #     # generate conv filters for classification layer
    #     task_encoding = self.encoding_task(task_id)
    #     task_encoding.unsqueeze_(2).unsqueeze_(2).unsqueeze_(2)

    #     x_feat = self.GAP(x)
    #     # print(x_feat.size(), task_encoding.size())
    #     x_cond = torch.cat([x_feat, task_encoding], 1)
    #     params = self.controller(x_cond)
    #     params.squeeze_(-1).squeeze_(-1).squeeze_(-1)

    #     # x8
    #     x = self.upsamplex2(x)
    #     x = x + skip3
    #     x = self.x8_resb(x)

    #     # x4
    #     x = self.upsamplex2(x)
    #     x = x + skip2
    #     x = self.x4_resb(x)

    #     # x2
    #     x = self.upsamplex2(x)
    #     x = x + skip1
    #     x = self.x2_resb(x)

    #     # x1
    #     x = self.upsamplex2(x)
    #     x = x + skip0
    #     x = self.x1_resb(x)

    #     head_inputs = self.precls_conv(x)

    #     N, _, D, H, W = head_inputs.size()
    #     head_inputs = head_inputs.reshape(1, -1, D, H, W)

    #     weight_nums, bias_nums = [], []
    #     weight_nums.append(8*8)
    #     weight_nums.append(8*8)
    #     weight_nums.append(8*1)
    #     bias_nums.append(8)
    #     bias_nums.append(8)
    #     bias_nums.append(1)
    #     weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)

    #     logits = self.heads_forward(head_inputs, weights, biases, N)
    #     logits = logits.reshape(-1, 1, D, H, W)

    #     return logits

#----------------------------------------------------------------------------------------
    def get_shared_features(self, input):
        # encoder
        x = self.conv1(input)
        x = self.layer0(x)
        skip0 = x
        x = self.layer1(x)
        skip1 = x
        x = self.layer2(x)
        skip2 = x
        x = self.layer3(x)
        skip3 = x
        x = self.layer4(x)
        x = self.fusionConv(x)
        x_feat = self.GAP(x)

        # decoder
        x = self.upsamplex2(x)
        x = x + skip3
        x = self.x8_resb(x)
        x = self.upsamplex2(x)
        x = x + skip2
        x = self.x4_resb(x)
        x = self.upsamplex2(x)
        x = x + skip1
        x = self.x2_resb(x)
        x = self.upsamplex2(x)
        x = x + skip0
        x = self.x1_resb(x)
        head_inputs = self.precls_conv(x)
        self.N, _, self.D, self.H, self.W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, self.D, self.H, self.W)
        return x_feat, head_inputs

    def get_dyn_params(self, x_feat, task_id):
        task_encoding = self.encoding_task(task_id)
        # print(task_encoding, task_encoding.size())
        task_encoding.unsqueeze_(2).unsqueeze_(2).unsqueeze_(2)
        x_cond = torch.cat([x_feat, task_encoding], 1)
        params = self.controller(x_cond)
        params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*2)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(2)
        weights, biases = self.parse_dynamic_params(params, 8, weight_nums, bias_nums)
        return weights, biases

    def forward(self, input, task_id):

        x_feat, head_inputs = self.get_shared_features(input)

        if self.training: 
            weights, biases = self.get_dyn_params(x_feat, task_id)
            logits = self.heads_forward(head_inputs, weights, biases, self.N)
            logits = logits.reshape(-1, 2, self.D, self.H, self.W)        
            return logits
        else:
            pred = [torch.zeros((self.N, self.D, self.H, self.W)).cuda()]
            for idx in range(task_id.size()[0]):
                current_task_id = task_id[idx].unsqueeze(0)
                current_task_id = current_task_id.expand(x_feat.size()[0])
                weights, biases = self.get_dyn_params(x_feat, current_task_id)
                logits = self.heads_forward(head_inputs, weights, biases, self.N)
                logits = logits.reshape(-1, 2, self.D, self.H, self.W)
                logits = logits[:, 1, ...]
                pred.append(logits)

            pred = torch.stack(pred, 1)
            return pred


def DoDNet(num_classes=1, weight_std=True):
    model = unet3D([1, 2, 2, 2, 2], num_classes, weight_std)
    return model


def testing():
    # data = torch.ones((2, 1, 128, 128, 128)).cuda()
    # code = torch.Tensor([1, 3]).long()
    model = DoDNet(1, True).cuda()

    # model.train()
    # output = model(data,code)

    from monai.inferers import sliding_window_inference
    from time import time
    
    model.eval()
    start = time()

    with torch.no_grad():
        output = sliding_window_inference(
            inputs=torch.ones((1, 1, 250, 250, 214)).cuda(), 
            roi_size=(128, 128, 128), 
            sw_batch_size=6, 
            predictor=model,
            overlap=0.5,
            task_id=torch.Tensor([0, 1, 2, 3, 4, 5]).long())

    print(f'inference time for 7 classes (sw_batch_size=8): {time() - start:.2f}')
    print(output.size())

# if __name__ == "__main__":
#     testing()
    
    