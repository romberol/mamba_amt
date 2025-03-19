import pytorch_lightning as pl
from torch.utils.data import DataLoader
from mamba_amt.data import MAESTRO
from mamba_amt.models import Mamba_AMT
from pytorch_lightning.callbacks import LearningRateMonitor
import json

if __name__ == "__main__":
    config = json.load(open("mamba_amt/configs/mamba_amt.json", "r"))
    model_config = config['model']
    training_config = config['training']

    wandb_logger = pl.loggers.WandbLogger(project="piano-transcription")

    train_dataset = MAESTRO(path="datasets/maestro-v3.0.0", sequence_length=training_config["sequence_length"], groups=['2006', '2009'], augment=False)
    val_dataset = MAESTRO(path="datasets/maestro-v3.0.0", sequence_length=training_config["sequence_length"], groups=['2008'])
    train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size=training_config["batch_size"], shuffle=False)
    
    # checkpoint_path = "piano-transcription/61pnybxp/checkpoints/epoch=399-step=3200.ckpt"
    checkpoint_path = None
    model = Mamba_AMT(model_config, lr=training_config["lr"])

    ckpt_callback = pl.callbacks.ModelCheckpoint(every_n_train_steps=len(train_loader) * training_config["save_every_n_epochs"], 
                                                 save_top_k=1)
    
    trainer = pl.Trainer(max_epochs=training_config["num_epochs"], logger=wandb_logger, log_every_n_steps=1,
                         callbacks=[LearningRateMonitor(logging_interval='step'), ckpt_callback])
    
    trainer.fit(model, train_loader, val_dataset, ckpt_path=checkpoint_path)
