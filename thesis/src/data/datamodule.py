import pytorch_lightning as pl
import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
import numpy as np
import datasets
from datasets import dataset_dict
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler
import re
from thesis.src.utils.collate import DataCollatorForT5MLM
from thesis.src.utils import collate
from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Any
import thesis.src.utils.utils as utils
import datasets
from thesis.src.utils.constants import (
    MODELS,
    XGLUE_TASKS,
    AIC_TASKS,
    AIC_PREFIX,
    XNLI_LANGS,
)


@dataclass
class DataModule(pl.LightningDataModule):
    model_name: Any
    size: Any = None
    to_text: bool = False
    task_names: Any = None
    t: int = 1
    batch_size: int = 32
    distributed: bool = False
    max_length_padding: int = 512
    mix_mlm: bool = False
    zero_shot_ctk: bool = False
    heterogenous_distributed: bool = True
    insert_prefix: bool = True
    init_seed: int = 42
    mlm_prob: float = 0.01

    def __post_init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODELS[self.model_name]["model_name"]
        )
        self.rng = np.random.default_rng(seed=self.init_seed)
        self.rng_integers_range = 10000  # magic

    def prepare_data(self):
        task_dict = self.load_data(self.task_names)
        n_classes, label2id_dict = self.get_label_info(task_dict)

        # for caching purposes
        self.preprocess_tasks(task_dict)

        # for logging
        batch_name_map_eval = self.get_task_mapping_split(task_dict, "validation")
        batch_name_map_test = self.get_task_mapping_split(task_dict, "test")

        # for xlm-r classification heads initialization
        #TODO can be deleted in favour of label2id_dict
        tasks = list(zip(list(task_dict), n_classes))

        return batch_name_map_eval, batch_name_map_test, tasks, label2id_dict

    def setup(self, stage: str):
        task_dict = self.load_data(self.task_names)
        task_dict = self.preprocess_tasks(task_dict)
        
        # for testing purposes
        if self.size:
            task_dict = self.cut_datasets(task_dict, self.size)

        if stage == "fit":
            # I need to set these here because dataloader functions don't take any args
            self.train = self.get_datasets_by_split(task_dict, "train")
            self.probs = self.proportional_probs(self.train)
            if self.mix_mlm:
                #TODO disgusting, it's not even effective, why am I keeping it?
                self.mlm = next(
                    self.mlm_dataset_iter(
                        np.sum(self.get_lengths(self.train)),
                        self.get_languages(task_dict),
                        chunks=self.trainer.max_epochs if self.trainer else 10,
                    )
                )
            self.eval = self.get_datasets_by_split(task_dict, "validation")

        elif stage == "validate":
            self.eval = self.get_datasets_by_split(task_dict, "validation")
        elif stage == "test" or stage == "predict":
            self.test = self.get_datasets_by_split(task_dict, "test")

        
    def train_dataloader(self):
        train = self.train
        probs = self.probs

        seed_epoch = self.trainer.current_epoch if self.trainer else self.init_seed

        if self.distributed and self.heterogenous_distributed and self.trainer:
            seed_local = (
                self.rng.integers(self.rng_integers_range, size=1)[0]
                + self.trainer.local_rank
            )
        else:
            seed_local = 0

        seed = seed_epoch + seed_local

        probs = self.temperature(probs, t=self.t)

        if self.mix_mlm:
            mlm_data = self.mlm
            probs = np.append(probs * (1 - self.mlm_prob), self.mlm_prob)
            train["mc4"] = mlm_data

        loaders = self.loaders_by_split(train, shuffle=True, seed=seed_epoch)
        return MultitaskDataloader(loaders, probs=probs, seed=seed)

    def val_dataloader(self):
        return self.loaders_by_split(self.eval)

    def test_dataloader(self):
        return self.loaders_by_split(self.test)

    def predict_dataloader(self):
        return self.loaders_by_split(self.test)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        pass

    def preprocess_tasks(self, task_dict):
        for name, data in task_dict.items():
            task_dict[name] = self.preprocess_task(name, data)
        return task_dict
    
    def preprocess_task(self, task_name, data):
        data = utils.format_columns(
            data,
            task_name,
            to_text=self.to_text,
            zero_shot_ctk=self.zero_shot_ctk,
            insert_prefix=self.insert_prefix,
        )
        data = utils.tokenize_data(
            self.tokenizer,
            data,
            task_name,
            to_text=self.to_text,
            max_length=self.max_length_padding,
        )
        return data

    def loaders_by_split(self, task_dict, shuffle=False, seed=42):
        loader_list = []
        for task_name, task in task_dict.items():
            for split_name, data in task.items():
                loader = self.get_dataloaders(data, shuffle=shuffle, seed=seed)
                named_loader = DataLoaderWithTaskname(task_name, split_name, loader)
                loader_list.append(named_loader)
        return loader_list

    def get_datasets_by_split(self, task_dict, split):
        ret = {}
        for name, task in task_dict.items():
            if name not in ret:
                ret[name] = dataset_dict.DatasetDict({})

            for task_split in task:
                if split not in task_split:
                    continue
                ret[name][task_split] = task[task_split]
        return ret
    def get_label_info(self, task_dict):
        label2id_dict = {}
        n_classes = []
        for task_name, data in task_dict.items():
            data_formatted = utils.format_columns(
                data,
                task_name,
                to_text=self.to_text,
                zero_shot_ctk=self.zero_shot_ctk,
                insert_prefix=self.insert_prefix,
            )

            label_names = data_formatted["train"].features["label"].names
            n_classes.append(len(label_names))
            label2id_dict[task_name] = {
                label: float(i) for i, label in enumerate(label_names)
            }
        return n_classes, label2id_dict

    def get_task_mapping_split(self, task_dict, split):
        task_name_map = []
        for task_name, task in task_dict.items():
            for split_name in filter(lambda x: split in x, task):
                task_name_map.append(task_name + "-" + split_name)
        return task_name_map

    def mlm_dataset_iter(self, length_overall, languages, chunks=1):
        length_by_language = (0.01 * length_overall / 0.99) // len(languages)
        mlm_iterators_data = {
            lang: load_dataset("mc4", lang, streaming=True, split="train")
            for lang in languages
        }
        for _ in range(chunks):
            yield self.mlm_chunk(mlm_iterators_data, languages, length_by_language)

    def mlm_chunk(self, mlm_iterators_data, languages, length_by_language):
        collator = DataCollatorForT5MLM(
            self.tokenizer,
            0.15,
            3.0,
            self.max_length_padding,
            self.targets_len,
            self.tokenizer.pad_token_id,
            self.tokenizer.pad_token_id,
        )
        mlm_data = {}
        for lang in languages:
            for i, entry in enumerate((mlm_iterators_data[lang])):
                if i >= max(0, length_by_language):
                    break
                if lang not in mlm_data:
                    mlm_data[lang] = []

                mlm_data[lang].append(entry)
            mlm_data[lang] = datasets.Dataset.from_list(mlm_data[lang])

        tokenized_dict_data = {
            lang: mlm_data[lang].map(
                self.tokenize_mix_function,
                remove_columns=["text", "url", "timestamp"],
                batched=True,
                load_from_cache_file=True,
            )
            for lang in mlm_data
        }


        (self.inputs_len, self.targets_len) = collator.compute_input_and_target_lengths(
            self.max_length_padding, 0.15, 3.0
        )

        lang_dict_data_grouped = {
            lang: tokenized_dict_data[lang].map(
                lambda x: collator.group_texts(self.inputs_len, x),
                batched=True,
                load_from_cache_file=True,
            )
            for lang in tokenized_dict_data
        }

        for k, v in lang_dict_data_grouped.items():
            v.set_format("np")

        mix_interleaved = interleave_datasets(
            list(lang_dict_data_grouped.values()),
            probabilities=[1 / len(languages)] * len(languages),
            seed=self.init_seed,
        )
        mix_interleaved = mix_interleaved.shuffle(seed=self.init_seed)
        mix_interleaved.set_format("np")
        mlm_dataset = dataset_dict.DatasetDict(
            {"train": datasets.Dataset.from_dict(collator(mix_interleaved))}
        )
        mlm_dataset.set_format("pt")
        return mlm_dataset

    def proportional_probs(self, task_dict):
        lengths = self.get_lengths(task_dict)
        probs = lengths / np.sum(lengths)
        return probs

    def get_lengths(self, task_dict):
        task_list = [split for task in task_dict.values() for split in task.values()]
        return np.array([len(split) for split in task_list])

    def load_data(self, task_names):
        tasks = {name : data for name, data in map(self.load_task, task_names)}
        return tasks

    def load_task(self, name):
        if name in XGLUE_TASKS:
            return (name, load_dataset("xglue", name))
        elif name in AIC_TASKS:
            return (name, load_dataset(AIC_PREFIX + name))
        elif name[-2:] in XNLI_LANGS and name[:-3] == "xnli":
            return (name, load_dataset("xnli", name[-2:]))
        elif name == "xnli_all":
            lang_sets = [
                load_dataset("xnli", lang, split="train").shuffle(seed=self.init_seed)
                for lang in XNLI_LANGS
            ]
            len_init = int(np.floor(len(lang_sets[0]) / len(lang_sets)))
            for i, lang_set in enumerate(lang_sets):
                lang_sets[i] = lang_set.select(range(i * len_init, (i + 1) * len_init))
            new_data = interleave_datasets(lang_sets, seed=self.init_seed)
            val_test = load_dataset("xglue", "xnli")
            val_test.pop("train")
            val_test["train"] = new_data
            return ("xnli", val_test)
        else:
            return (name, load_dataset(name))

    def cut_datasets(self, task_dict, size):
        for name in task_dict:
            task_dict[name]["train"] = task_dict[name]["train"].select(np.arange(size))
        return task_dict

    def get_languages(self, task_dict):
        languages = []
        for split in [split for task_data in task_dict.values() for split in task_data]:
            if "validation." in split:
                languages.append(re.findall("(?:validation.)(.{2})", split)[0])
        languages = np.unique(languages)
        return languages

    def prepare_mlm(self, mlm_iterators_data, languages, mlm_len_per_lang, chunks=1):
        mlm_data = {}
        for lang in languages:
            for i, entry in enumerate((mlm_iterators_data[lang])):
                if i >= max(0, mlm_len_per_lang):
                    break
                if lang not in mlm_data:
                    mlm_data[lang] = []

                mlm_data[lang].append(entry)
            mlm_data[lang] = datasets.Dataset.from_list(mlm_data[lang])

        tokenized_dict_data = {
            lang: mlm_data[lang].map(
                self.tokenize_mix_function,
                remove_columns=["text", "url", "timestamp"],
                batched=True,
                load_from_cache_file=True,
            )
            for lang in mlm_data
        }


        (self.inputs_len, self.targets_len) = collate.compute_input_and_target_lengths(
            self.max_length_padding, 0.15, 3.0
        )

        lang_dict_data_grouped = {
            lang: tokenized_dict_data[lang].map(
                lambda x: collate.group_texts(self.inputs_len, x),
                batched=True,
                load_from_cache_file=True,
            )
            for lang in tokenized_dict_data
        }

        for k, v in lang_dict_data_grouped.items():
            v.set_format("np")
        return lang_dict_data_grouped

    def tokenize_mix_function(self, examples):
        return self.tokenizer(examples["text"], return_attention_mask=False)


    def get_named_dataloaders(self, loader_dict, task_name):
        named_loaders = {}
        for name, loader in loader_dict.items():
            named_loaders[name] = DataLoaderWithTaskname(task_name, name, loader)
        return named_loaders

    def get_dataloaders(
        self, task, shuffle=False, collator=None, drop_last=False, seed=42
    ):
        if self.distributed:
            sampler = DistributedSampler(
                task,
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        else:
            generator = torch.Generator()
            generator = generator.manual_seed(seed)
            sampler = RandomSampler(task, generator=generator)

        return DataLoader(
            dataset=task,
            batch_size=self.batch_size,
            collate_fn=collator,
            # shuffle=shuffle if not self.distributed else None,
            drop_last=drop_last,
            sampler=sampler,
        )

    def tokenize_map(self, tokenizer, column_names, example):
        return tokenizer(
            *[example[column_name] for column_name in column_names],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

    def temperature(self, probs, t):
        temp_probs = probs ** (1 / t)
        normalized_temp = temp_probs / np.sum(temp_probs)
        return normalized_temp


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
