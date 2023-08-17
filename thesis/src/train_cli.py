from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from thesis.src.models.modelmodule import ModelModule
from thesis.src.data.datamodule import DataModule

class MtlLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        if self.subcommand:
            config = self.config[self.subcommand]
        data = self.datamodule_class(**config["data"].as_dict())
        for arg, val in data.prepare_data().items():
            config["model"][arg] = val
        
def main():
    cli = MtlLightningCLI(model_class = ModelModule,
                       datamodule_class = DataModule)
if __name__ == "__main__":
    main()