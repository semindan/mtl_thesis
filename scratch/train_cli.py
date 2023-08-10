from lightning.pytorch.cli import LightningCLI
from thesis.src.models.modelmodule import ModelModule
from thesis.src.models.mt5.mt5_module import MT5
cli = LightningCLI(ModelModule)
