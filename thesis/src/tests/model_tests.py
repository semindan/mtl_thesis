import unittest
from thesis.src.data.datamodule import DataModule
from thesis.src.models.modelmodule import ModelModule
from thesis.src.utils import utils
from pprint import pprint
import torch

class TestDataModule(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model_name = "mt5"
        self.task_names = ["paws-x", "nc", "wpr"]

    def test_one_train_step(self):
        data_module = DataModule(model_name = self.model_name, task_names=self.task_names, size=10, batch_size=4)
        arg_dict = data_module.prepare_data()
        model = ModelModule(model_name=self.model_name, **arg_dict)
        data_module.setup("fit")
        dataloader = data_module.train_dataloader()
        batch = next(iter(dataloader))
        loss = model.training_step(batch, 0)
        self.assertFalse(loss.isnan(), "loss must not be nan")

if __name__ == '__main__':
    unittest.main()