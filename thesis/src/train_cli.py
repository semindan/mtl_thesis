from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from thesis.src.models.modelmodule import ModelModule
from thesis.src.data.datamodule import DataModule
import torch

class MtlLightningCLI(LightningCLI):
    def before_instantiate_classes(self) -> None:
        config = self.config[self.subcommand] if self.subcommand else self.config
        data = self.datamodule_class(**config["data"].as_dict())
        for arg, val in data.prepare_data().items():
            config["model"][arg] = val
        
def set_precision():
    torch.set_float32_matmul_precision("medium")

def main():
    set_precision()
    cli = MtlLightningCLI(model_class = ModelModule,
                       datamodule_class = DataModule, save_config_kwargs={"overwrite" : True})

if __name__ == "__main__":
    main()
