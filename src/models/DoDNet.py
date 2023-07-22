#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author      : Han Liu
# Date Created: 07/26/2022


import torch.nn as nn
from torch.nn import functional as F
import torch

in_place = True


# DoDNet implementation adapted from https://github.com/jianpengz/DoDNet/blob/main/a_DynConv/unet3D_DynConv882.py


class unet3D(nn.Module):
    def __init__(self, num_classes=3):
        super(unet3D, self).__init__()
        self.num_classes = num_classes
        self.norm_layer=nn.BatchNorm3d
        nbase = 8
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

        self.precls_conv = nn.Sequential(
            nn.ReLU(inplace=in_place),
            nn.Conv3d(8, 8, kernel_size=1))

        self.GAP = nn.Sequential(
            nn.ReLU(inplace=in_place),
            torch.nn.AdaptiveAvgPool3d((1,1,1)))

        self.controller = nn.Conv3d(16*nbase+(self.num_classes-1), 162, kernel_size=1, stride=1, padding=0) 


    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=True, bin_selu=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU(),
                self.norm_layer(out_channels))
        elif bin_selu:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.SELU())
        else:
            layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        return layer


    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, self.num_classes-1))
        for i in range(N):
            task_encoding[i, task_id[i]]=1                                                 
        return task_encoding.cuda()


    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)
        num_insts = params.size(0)
        num_layers = len(weight_nums)
        params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))
        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]
        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)
        return weight_splits, bias_splits


    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv3d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x


    def get_shared_features(self, input):
        syn1 = self.conv1_1(input)
        syn2 = self.conv2_1(self.conv1_2(syn1))
        syn3 = self.conv3_1(self.conv2_2(syn2))
        syn4 = self.conv4_1(self.conv3_2(syn3))
        x = self.conv5(self.conv4_2(syn4))
        x = self.conv6(x)
        x_feat = self.GAP(x)
        x = self.conv7_2(torch.cat([self.upsample(x), syn4], 1))
        x = self.conv8_2(torch.cat([self.upsample(x), syn3], 1))
        x = self.conv9_2(torch.cat([self.upsample(x), syn2], 1))
        x = self.conv10_2(torch.cat([self.upsample(x), syn1], 1))
        head_inputs = self.precls_conv(x)
        self.N, _, self.D, self.H, self.W = head_inputs.size()
        head_inputs = head_inputs.reshape(1, -1, self.D, self.H, self.W)
        return x_feat, head_inputs

        # # encoder
        # x = self.conv1(input)
        # x = self.layer0(x)
        # skip0 = x
        # x = self.layer1(x)
        # skip1 = x
        # x = self.layer2(x)
        # skip2 = x
        # x = self.layer3(x)
        # skip3 = x
        # x = self.layer4(x)
        # x = self.fusionConv(x)
        # x_feat = self.GAP(x)

        # # decoder
        # x = self.upsamplex2(x)
        # x = x + skip3
        # x = self.x8_resb(x)
        # x = self.upsamplex2(x)
        # x = x + skip2
        # x = self.x4_resb(x)
        # x = self.upsamplex2(x)
        # x = x + skip1
        # x = self.x2_resb(x)
        # x = self.upsamplex2(x)
        # x = x + skip0
        # x = self.x1_resb(x)
        # head_inputs = self.precls_conv(x)
        # self.N, _, self.D, self.H, self.W = head_inputs.size()
        # head_inputs = head_inputs.reshape(1, -1, self.D, self.H, self.W)
        # return x_feat, head_inputs

    def get_dyn_params(self, x_feat, task_id):
        task_encoding = self.encoding_task(task_id)
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


def DoDNet(num_classes=1):
    model = unet3D(num_classes)
    return model


def testing():
    data = torch.ones((2, 1, 128, 128, 128)).cuda()
    code = torch.Tensor([1, 3]).long()
    model = DoDNet(8).cuda()

    model.train()
    output = model(data,code)
    print(output.shape)

    # from monai.inferers import sliding_window_inference
    # from time import time
    
    # model.eval()
    # start = time()

    # with torch.no_grad():
    #     output = sliding_window_inference(
    #         inputs=torch.ones((1, 1, 250, 250, 214)).cuda(), 
    #         roi_size=(128, 128, 128), 
    #         sw_batch_size=6, 
    #         predictor=model,
    #         overlap=0.5,
    #         task_id=torch.Tensor([0, 1, 2, 3, 4, 5]).long())

    # print(f'inference time for 7 classes (sw_batch_size=8): {time() - start:.2f}')
    # print(output.size())



# if __name__ == "__main__":
#     testing()
    
    