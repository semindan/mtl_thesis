import numpy as np

class StrIgnoreDevice(str):
    def to(self, device):
        return self

class DataLoaderWithTaskname:
    def __init__(self, task_name, split_name, data_loader):
        self.task_name = task_name
        self.split_name = split_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            batch["split_name"] = StrIgnoreDevice(self.split_name)
            batch["task_size"] = StrIgnoreDevice(len(self.dataset))
            yield batch


class MultitaskDataloader:
    def __init__(self, dataloader_list, probs, seed=42):
        self.dataloader_list = dataloader_list
        self.num_batches_list = [len(dataloader) for dataloader in self.dataloader_list]
        self.dataset = [None] * sum(
            len(dataloader.dataset) for dataloader in self.dataloader_list
        )
        self.probs = probs
        self.seed = seed

    def __len__(self):
        return sum(self.num_batches_list)

    def __iter__(self):
        task_choice_list = []
        probs_list = []
        for i, _ in enumerate(self.dataloader_list):
            task_choice_list += [i] * self.num_batches_list[i]
            probs_list += [
                self.probs[i] / self.num_batches_list[i]
            ] * self.num_batches_list[i]

        rng = np.random.default_rng(seed=self.seed)
        task_choice_list = rng.choice(
            task_choice_list, len(task_choice_list), p=probs_list, replace=False
        )
        dataloader_iter_list = [iter(dataloader) for dataloader in self.dataloader_list]

        for task_choice in task_choice_list:
            next_batch = next(dataloader_iter_list[task_choice])
            yield next_batch