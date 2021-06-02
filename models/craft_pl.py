import torch
import numpy as np
import cv2
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models
from collections import namedtuple

import pytorch_lightning as pl

from utils.data_manipulation import resize, normalize_mean_variance
from utils.craft_utils import hard_negative_mining, Heatmap2Box
from utils.misc import calculate_batch_fscore, generate_word_bbox_batch


class CRAFT(pl.LightningModule):
    def __init__(self, cfg, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()
        self.cfg = cfg
        
        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        # ToDo - Remove the interpolation and make changes in the dataloader to make target width, height //2

        y = F.interpolate(y, size=(768, 768), mode='bilinear', align_corners=False)

        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer

    def training_step(self, batch, batch_num):
        image, weight, weight_affinity, _ = batch

        output = self(image)
        loss = self.cal_loss(output, weight, weight_affinity)
        self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_num):
        image, weight, weight_affinity, _ = batch

        output = self(image)
        loss = self.cal_loss(output, weight, weight_affinity)

        if type(output) == list:
            output = torch.cat(output, dim=0)

        predicted_bbox = generate_word_bbox_batch(
            output[:, 0, :, :].data.cpu().numpy(),
            output[:, 1, :, :].data.cpu().numpy(),
            character_threshold=self.cfg.THRESHOLD_CHARACTER,
            affinity_threshold=self.cfg.THRESHOLD_AFFINITY,
            word_threshold=self.cfg.THRESHOLD_WORD,
        )

        target_bbox = generate_word_bbox_batch(
            weight.data.cpu().numpy(),
            weight_affinity.data.cpu().numpy(),
            character_threshold=self.cfg.THRESHOLD_CHARACTER,
            affinity_threshold=self.cfg.THRESHOLD_AFFINITY,
            word_threshold=self.cfg.THRESHOLD_WORD
        )

        fscore, precision, recall = calculate_batch_fscore(
            predicted_bbox,
            target_bbox,
            threshold=self.cfg.THRESHOLD_FSCORE,
            text_target=None
        )


        return {'val_loss':loss, 'fscore':fscore,
                'precision':precision, 'recall':recall}

    def validation_epoch_end(self, outputs):
        val_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        fscore = sum([x['fscore'] for x in outputs]) / len(outputs)
        precision = sum([x['precision'] for x in outputs]) / len(outputs)
        recall = sum([x['recall'] for x in outputs]) / len(outputs)

        self.log('val_loss', val_loss)
        self.log('fscore', fscore)
        self.log('precision', precision)
        self.log('recall', recall)

    def cal_loss(self, output, character_map, affinity_map):
        """
        :param output: prediction output of the model of shape [batch_size, 2, height, width]
        :param character_map: target character map of shape [batch_size, height, width]
        :param affinity_map: target affinity map of shape [batch_size, height, width]
        :return: loss containing loss of character heat map and affinity heat map reconstruction
        """

        batch_size, channels, height, width = output.shape

        output = output.permute(0, 2, 3, 1).contiguous().view([batch_size * height * width, channels])

        character = output[:, 0]
        affinity = output[:, 1]

        affinity_map = affinity_map.view([batch_size * height * width])
        character_map = character_map.view([batch_size * height * width])

        loss_character = hard_negative_mining(character, character_map, self.cfg)
        loss_affinity = hard_negative_mining(affinity, affinity_map, self.cfg)

        all_loss = loss_character * 2 + loss_affinity

        return all_loss
    
    def preprocessImage(self, image, side=768):
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], repeats=3, axis=2)
        elif image.shape[2] == 1:
            image = np.repeat(image, repeats=3, axis=2)
        else:
            image = image[:, :, 0: 3]

        target_size = (768, 768)  # (w, h)
        color=(114, 114, 114)

        height, width, channel = image.shape
        
        if self.cfg.PAD:
            ratio = min(target_size[1] / height, target_size[0] / width)

            image = cv2.resize(image, (int(width*ratio), int(height*ratio)), interpolation=cv2.INTER_CUBIC)

            dw = (target_size[0] - image.shape[1]) / 2
            dh = (target_size[1] - image.shape[0]) / 2

            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        else:
            image = cv2.resize(image, (side, side), interpolation=cv2.INTER_CUBIC)
            
        image = normalize_mean_variance(image).transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).unsqueeze(0)

        return image, (width, height)
    
    def get_boxes(self, output, resize_info):
        scoreText = output[0][0, :, :].data.cpu().numpy()
        scoreLink = output[0][1, :, :].data.cpu().numpy()

        heatmap2BoxArgs = [scoreText, scoreLink, resize_info]

        return Heatmap2Box(heatmap2BoxArgs, self.cfg)
        

class vgg16_bn(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super(vgg16_bn, self).__init__()
        vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(12):         # conv2_2
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):         # conv3_3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):         # conv4_3
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):         # conv5_3
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

         # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():      # only first conv
                param.requires_grad= False

    def forward(self, X):
        h = self.slice1(X)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()