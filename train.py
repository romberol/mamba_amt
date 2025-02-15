import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mamba_amt.data import MAESTRO
from mamba_amt.models import AMT_Trainer

# Hyperparameters
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 100

if __name__ == "__main__":
    wandb_logger = pl.loggers.WandbLogger(project="piano-transcription")

    dataset = MAESTRO(path="datasets/maestro-2004", sequence_length=327680, groups=['2004'])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model_config = {
        'mamba_blocks': 1,  # number of mamba blocks
        'd_model': 256,     # model dimension
        'd_state': 64,      # B anc C dimensions
        'd_conv': 4,        # local convolution width
        'expand': 2 ,       # block expansion factor
        'out_features': 88
    }

    model = AMT_Trainer(model_config, lr=LR)
    trainer = pl.Trainer(max_epochs=EPOCHS, logger=wandb_logger, log_every_n_steps=1)
    
    trainer.fit(model, loader)


