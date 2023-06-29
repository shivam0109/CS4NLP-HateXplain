import pytorch_lightning as pl

from dataset import HateXplainDataModule
from model import HateXPlainTransformer

def main():
    pl.seed_everything(42)

    datamodule = HateXplainDataModule()
    datamodule.prepare_data()
    datamodule.setup("fit")

    transformer = HateXPlainTransformer()

    trainer = pl.Trainer(
        max_epochs=1,
    )

    trainer.fit(transformer, datamodule=datamodule)

if __name__ == "__main__":
    main()