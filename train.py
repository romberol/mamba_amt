import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mamba_amt.data import MAESTRO
from mamba_amt.models import AMT_Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

# Hyperparameters
BATCH_SIZE = 16
LR = 5e-4
EPOCHS = 200

if __name__ == "__main__":
    wandb_logger = pl.loggers.WandbLogger(project="piano-transcription")

    train_dataset = MAESTRO(path="datasets/maestro-v3.0.0", sequence_length=327680, groups=['2006'], augment=False)
    val_dataset = MAESTRO(path="datasets/maestro-v3.0.0", sequence_length=327680, groups=['2008'])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model_config = {
        'mamba_blocks': 1,  # number of mamba blocks
        'd_model': 256,     # model dimension
        'd_state': 64,      # B anc C dimensions
        'd_conv': 4,        # local convolution width
        'expand': 2 ,       # block expansion factor
        'out_features': 88
    }

    # checkpoint_path = "piano-transcription/ps0pdb01/checkpoints/epoch=199-step=1600.ckpt"
    checkpoint_path = None
    model = AMT_Trainer(model_config, lr=LR)
    trainer = pl.Trainer(max_epochs=EPOCHS, logger=wandb_logger, log_every_n_steps=1,
                         callbacks=[LearningRateMonitor(logging_interval='step')])
    
    trainer.fit(model, train_loader, val_dataset, ckpt_path=checkpoint_path)
