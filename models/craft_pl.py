import torch
import pytorch_lightning as pl

from models.craft_model import CRAFT_
from utils.craft_utils import hard_negative_mining
from utils.misc import calculate_batch_fscore, generate_word_bbox_batch

class CRAFT(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = CRAFT_()
        
    def forward(self, images):
        output = self.model(images)
        return output
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer
        
    def training_step(self, batch, batch_num):
        image, weight, weight_affinity, _ = batch
        
        output = self.model(image)
        loss = self.cal_loss(output, weight, weight_affinity)
        self.log('train_loss', loss)
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_num):
        image, weight, weight_affinity, _ = batch
        
        output = self.model(image)
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
        
        self.log('val_loss', loss)
        self.log('fscore', fscore)
        self.log('precision', precision)
        self.log('recall', recall)
        
        return {'val_loss':loss, 'val_fscore': fscore}
    
    def validation_epoch_end(self, outputs):
        f_score = [x['val_fscore'] for x in outputs]
        avg_fscore = sum(f_score) / len(f_score)
        print(f"\nEpoch {self.current_epoch} | avg_fscore:{avg_fscore}\n")
    
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