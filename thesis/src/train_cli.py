from lightning.pytorch.cli import LightningCLI
from thesis.src.models.modelmodule import ModelModule
from thesis.src.data.datamodule import DataModule



def main():
    cli = LightningCLI(None,
                       datamodule_class = DataModule,
                       run=False)
    # cli.trainer.fit(cli.model, cli.datamodule)
if __name__ == "__main__":
    main()