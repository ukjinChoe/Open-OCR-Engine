import pytorch_lightning as pl

class DeepTextRecog(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, imgs):
        pass
        
    def configure_optimizers(self):
        pass
        
    def training_step(self, batch, batch_num):
        pass
    
    def validation_step(self, batch, batch_num):
        pass
    
    def validation_epoch_end(self, outputs):
        pass
    
    def cal_loss(self, logits, targets):
        pass