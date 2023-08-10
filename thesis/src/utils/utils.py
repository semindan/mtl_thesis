import torchmetrics
import torch.nn as nn
from datasets import Features
from thesis.src.utils.metrics import get_guids
from datasets.dataset_dict import DatasetDict
import os
import copy

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def format_columns(
    data, task_name, to_text=False, zero_shot_ctk=False, insert_prefix=False
):
    match task_name:
        case "xnli":
            data = format_xnli(data)
        case "wpr":
            #TODO why was it necessary to deepcopy???
            # I surely remember something went wrong, yet I don't remember what exactly
            # data = copy.deepcopy(data)
            data = DatasetDict({k: v.add_column(name="guids", column=get_guids(v, "query")) for k, v in data.items()})
            # for part in data:
            #     data[part] = data[part].add_column(
            #         name="guids", column=get_guids(data[part], "query")
            #     )
            data = format_wpr(data, to_text=to_text)
        case "paws-x":
            data = format_paws_x(data)
        case "qadsm":
            data = format_qadsm(data, to_text=to_text)
        case "qam":
            data = format_qam(data)
        case "nc":
            data = format_nc(data, to_text=to_text)
        case "ctkfacts_nli":
            data = format_ctk(data, zero_shot=zero_shot_ctk)
        case other:
            data = data

    if to_text:
        column_names = list(
            filter(lambda x: x != "label" and x != "guids", data["train"].column_names)
        )
        label_names = data["train"].features["label"].names

        if zero_shot_ctk and "ctk" in task_name:
            task_name = "xnli"

        def data_to_text(example):
            string = (task_name + ":") if insert_prefix else ""
            for column_name in column_names:
                string += " " + column_name + ": " + example[column_name]
            example["input"] = string
            example["target"] = label_names[example["label"]]
            if task_name == "wpr":
                example["guids"] = example["guids"]
            # example["label_id"] = example["label"]
            return example

        data = data.map(data_to_text, batched=False)
        # data = data.rename_column("label", "label_id")
        data = data.remove_columns(column_names=column_names)

    data.set_format("pt")
    return data


def format_ctk(data, zero_shot=False):
    data = data.remove_columns(["id"])
    data = data.align_labels_with_mapping(
        {"SUPPORTS": 0, "NOT ENOUGH INFO": 1, "REFUTES": 2}, "label"
    )  # xnli style
    old_features = data["train"].features
    data = data.cast(
        Features(
            {
                "evidence": old_features["evidence"],
                "claim": old_features["claim"],
                "label": old_features["label"],
            }
        )
    )
    if zero_shot:
        data = data.rename_column("evidence", "premise")
        data = data.rename_column("claim", "hypothesis")
        for split in data:
            data[split].features["label"].names = [
                "entailment",
                "neutral",
                "contradiction",
            ]

    return data


def format_nc(data, to_text=False):
    if not to_text:
        data = data.map(
            lambda x: format_concat("news_title", "news_body", "news_text", x)
        )
        data = data.remove_columns(["news_title", "news_body"])
    data = data.rename_column("news_category", "label")
    return data


def format_qadsm(data, to_text=False):
    if not to_text:
        data = data.map(
            lambda x: format_concat("ad_title", "ad_description", "ad_text", x)
        )
        data = data.remove_columns(["ad_title", "ad_description"])
    data = data.rename_column("relevance_label", "label")
    return data


def format_wpr(data, to_text=False):
    if not to_text:
        data = data.map(
            lambda x: format_concat(
                "web_page_title", "web_page_snippet", "web_page_text", x
            )
        )
        data = data.remove_columns(["web_page_title", "web_page_snippet"])
    data = data.rename_column("relavance_label", "label")
    return data


def format_xnli(data):
    return data


def format_qam(data):
    return data


def format_paws_x(data):
    return data


def format_concat(old_1, old_2, new, example):
    example[new] = example[old_1] + " " + example[old_2]
    return example


def tokenize_data(tokenizer, data, task_name, to_text=False, max_length=512):
    column_names = list(
        filter(lambda x: x != "label" and x != "guids", data["train"].column_names)
    )
    if to_text:

        def tokenize(example):
            model_inputs = tokenizer(
                example["input"],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            labels = tokenizer(
                example["target"],
                padding="longest",
                max_length=128,
                truncation=True,
                return_tensors="pt",
            )
            labels_encoding = labels.input_ids
            labels_encoding[labels_encoding == tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels_encoding
            return model_inputs

        data = data.map(tokenize, batched=True)
        data = data.remove_columns(column_names=column_names)

    else:

        def tokenize(example):
            return tokenizer(
                *[example[column_name] for column_name in column_names],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

        data = data.map(tokenize, batched=True)
        data = data.remove_columns(column_names=column_names).rename_column(
            "label", "labels"
        )

    data.set_format("pt")
    return data
