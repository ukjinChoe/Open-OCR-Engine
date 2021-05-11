import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from hparams import cfg
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TRAIN OCR-ENGINE MODULE')
    parser.add_argument('-m', '--module', type=str,
                        choices=['detector', 'recognizer'],
                        help='module to train')
    parser.add_argument('-v', '--version', type=int, default=0,
                        help='version number')
    parser.add_argument('-bs', '--batch_size', type=int, default=4,
                        help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5,
                        help='learning rate for training [detector: 5e-5, recognizer: 1.0]')
    parser.add_argument('-e', '--max_epoch', type=int, default=100,
                        help='max epoch')
    parser.add_argument('-nw', '--num_workers', type=int, default=4,
                        help='number of workers for calling data')
    parser.add_argument('-rt', '--resume_training', type=str, default=None,
                        help='resume from certain checkpoint file')

    args = parser.parse_args()
    cfg.lr = args.learning_rate

    if args.module == 'detector':
        from models.craft_pl import CRAFT
        from datasets.craft_dataset import DatasetSYNTH
        dataset = DatasetSYNTH(cfg)
        model = CRAFT(cfg)
        collate = None
    else:
        from models.deepTextRecog_pl import DeepTextRecog
        from datasets.deepTextRecog_dataset import DatasetSYNTH
        dataset = DatasetSYNTH(cfg)
        model = DeepTextRecog(cfg, dataset.tokens)
        collate = model.collate

    trainSize = int(len(dataset)*0.9)
    trainDataset, validDataset = random_split(dataset, [trainSize, len(dataset)-trainSize])
    trainDataloader = DataLoader(trainDataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate)
    validDataloader = DataLoader(validDataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=collate)

    logger = TensorBoardLogger('tb_logs', name=args.module,
                               version=args.version, default_hp_metric=False)
    # lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        monitor='fscore' if args.module == 'detector' else 'acc',
        dirpath=f'checkpoints/version_{args.version}',
        filename='checkpoints-{epoch:02d}-{fscore:.2f}',
        save_top_k=3,
        mode='max',
    )

    n_gpu = torch.cuda.device_count()

    trainer = pl.Trainer(gpus=n_gpu, max_epochs=args.max_epoch, logger=logger,
                         num_sanity_val_steps=1, accelerator='ddp',
                         callbacks=[ckpt_callback],
                         plugins=DDPPlugin(find_unused_parameters=False),
                         resume_from_checkpoint=args.resume_training)

    trainer.fit(model, train_dataloader=trainDataloader, val_dataloaders=validDataloader)
