import pytorch_lightning as pl
import torch

import torch.nn.functional as F
from nltk.metrics.distance import edit_distance

from models.deepTextRecog_model import DeepTextRecog_
from datasets.deepTextRecog_dataset import AlignCollate
from utils.deepTextRecog_utils import CTCLabelConverter, AttnLabelConverter

class DeepTextRecog(pl.LightningModule):
    def __init__(self, cfg, tokens):
        super(DeepTextRecog, self).__init__()
        self.collate = AlignCollate(cfg)
        
        if 'CTC' in cfg.Prediction:
            self.converter = CTCLabelConverter(tokens)
            self.cal_loss = torch.nn.CTCLoss(zero_infinity=True)
        else:
            self.converter = AttnLabelConverter(tokens)
            self.cal_loss = torch.nn.CrossEntropyLoss(ignore_index=0)

        cfg.num_class = len(self.converter.tokens)
        
        self.model = DeepTextRecog_(cfg)
        self.cfg = cfg
        
    def forward(self, images):
        output = self.model(images, text=None, is_train=False)
        return output
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.cfg.lr)
        return optimizer
        
    def training_step(self, batch, batch_num):
        image_tensors, labels = batch
        text, length = self.converter.encode(
                labels, batch_max_length=self.cfg.batch_max_length)
        if 'CTC' in self.cfg.Prediction:
            output = self.model(image_tensors, text)
            output_size = torch.IntTensor([output.size(1)] * self.cfg.batch_size)
            output = output.log_softmax(2).permute(1, 0, 2)
            loss = self.cal_loss(output, text, output_size, length) / self.cfg.batch_size
        else:
            # align with Attention.forward
            output = self.model(image_tensors, text[:, :-1])
            target = text[:, 1:]
            loss = self.cal_loss(output.view(-1, output.shape[-1]), target.contiguous().view(-1))
            
        self.log('train_loss', loss)
            
        return {'loss': loss}
    
    def validation_step(self, batch, batch_num):
        image_tensors, labels = batch
        
        batch_size = image_tensors.size(0)
        length_of_data = batch_size
        # For max length prediction
        length_for_pred = torch.IntTensor([self.cfg.batch_max_length] * batch_size)
        text_for_pred = torch.LongTensor(batch_size, self.cfg.batch_max_length + 1).fill_(0)

        text_for_loss, length_for_loss = self.converter.encode(labels,
                                    batch_max_length=self.cfg.batch_max_length)
        
        if 'CTC' in self.cfg.Prediction:
            preds = self.model(image_tensors, text_for_pred)

            # Calculate evaluation loss for CTC deocder.
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            # permute 'preds' to use CTCloss format
            loss = self.cal_loss(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

            # Select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index.data, preds_size.data)
        
        else:
            preds = self.model(image_tensors, text_for_pred, is_train=False)

            preds = preds[:, :text_for_loss.shape[1] - 1, :]
            target = text_for_loss[:, 1:]  # without [GO] Symbol
            loss = self.cal_loss(preds.contiguous().view(-1, preds.shape[-1]), target.contiguous().view(-1))

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)
            labels = self.converter.decode(text_for_loss[:, 1:], length_for_loss)
            
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        
        n_correct, norm_ED = self.calculate_acc(labels, preds_str, preds_max_prob)

        acc = n_correct / length_of_data * 100
        norm_ED = norm_ED / float(length_of_data)
          
        return {'val_loss': loss, 'acc': acc, 'norm_ED': norm_ED}
    
    def validation_epoch_end(self, outputs):
        val_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        acc = sum([x['acc'] for x in outputs]) / len(outputs)
        norm_ED = sum([x['norm_ED'] for x in outputs]) / len(outputs)
        
        self.log('val_loss', val_loss)
        self.log('acc', acc)    
        self.log('norm_ED', norm_ED)
    
    def calculate_acc(self, labels, preds_str, preds_max_prob):
        confidence_score_list = []
        
        n_correct = 0
        norm_ED = 0
        
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if 'Attn' in self.cfg.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

            if pred == gt:
                n_correct += 1

            '''
            (old version) ICDAR2017 DOST Normalized Edit Distance https://rrc.cvc.uab.es/?ch=7&com=tasks
            "For each word we calculate the normalized edit distance to the length of the ground truth transcription."
            if len(gt) == 0:
                norm_ED += 1
            else:
                norm_ED += edit_distance(pred, gt) / len(gt)
            '''

            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)
            
        return n_correct, norm_ED