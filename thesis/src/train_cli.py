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
        m_cfg = config["model"]
        (m_cfg["batch_name_map_eval"],
         m_cfg["batch_name_map_test"],
         m_cfg["tasks"],
         m_cfg["label2id"]) = data.prepare_data()
        

def init_callbacks(args):
    # lr_monitor = LearningRateMonitor(logging_interval="step")
    # checkpoint_callback = ModelCheckpoint(
    #     # every_n_epochs=1,
    #     save_top_k=1,
    #     save_last=True,
    #     monitor="eval/accumulate",
    #     mode="max",
    #     dirpath=args.save_path + args.name,
    #     filename="best",
    #     save_on_train_epoch_end=False,
    # )
    # early_stop_callback = EarlyStopping(
    #     monitor="eval/accumulate",
    #     min_delta=0.0001,
    #     patience=35,
    #     verbose=False,
    #     mode="max",
    #     check_finite=True,
    # )
    # return [lr_monitor, checkpoint_callback, early_stop_callback]
    # return [LearningRateMonitor, ModelCheckpoint, EarlyStopping]
    return None

def main():
    
    cli = MtlLightningCLI(model_class = ModelModule,
                       datamodule_class = DataModule,
                       trainer_defaults = init_callbacks(None))
if __name__ == "__main__":
    main()