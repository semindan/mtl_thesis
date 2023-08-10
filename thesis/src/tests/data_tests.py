import unittest
from thesis.src.data.datamodule import DataModule
from thesis.src.utils import utils
from pprint import pprint

class TestDataModule(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.model_name = "mt5"
        self.task_names = ["paws-x", "nc", "wpr"]

    def test_task_names(self):
        data_module = DataModule(model_name = self.model_name, task_names=self.task_names)
        self.assertEqual(data_module.task_names, self.task_names, 'task names must match')

    def test_load_data(self):
        data_module = DataModule(model_name = self.model_name, task_names=self.task_names)
        data = data_module.load_data(self.task_names)
        self.assertEqual(sorted(data), sorted(self.task_names), 'task names must match')

    def test_load_wpr_guids(self):
        task_names = ["wpr"]
        data_module = DataModule(model_name = self.model_name, task_names=task_names)
        data = data_module.load_data(task_names)
        data = utils.format_columns(data[task_names[0]], task_names[0], to_text=True)
        self.assertEqual(len(data["train"].features), 4, 'guids must be added')
        

    def test_prepare_data(self):
        # data_module = DataModule(model_name = self.model_name, task_names=self.task_names)
        # data_module.prepare_data()
        # print(data_module.)
        pass


if __name__ == '__main__':
    unittest.main()