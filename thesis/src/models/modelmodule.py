# %%
from typing import Any, Dict
import pytorch_lightning as pl
import torch
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer
import thesis.src.utils.metrics as metrics
from thesis.src.models.mt5.mt5_module import MT5
from thesis.src.models.xlmr.xlmr_module import XLMRModule
from thesis.src.models.mt5.mt5_module_pcgrad import MT5PCGrad
from thesis.src.models.mbert.mbert_module import MBERTModule
from thesis.src.utils.constants import MODELS
import os

# def int_cast(word):
#     if len(word) > 1:
#         word = word[0]
#     return float(word) if word.isdigit() else -1.0


# def decode_sequences(tokenizer, sequences):
#     sequences[sequences == -100] = tokenizer.pad_token_id
#     return tokenizer.batch_decode(sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True)
# self.__post_init__().




class ModelModule(pl.LightningModule):
    def __init__(
        self,
        model_name,
        tasks,
        batch_name_map_eval,
        batch_name_map_test,
        label2id,
        peft=False,
        peft_checkpoint=False,
        num_accum_batches=1,
        r3f=False,
        r4f=False,
        scale_loss=False,
        retain_grads=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_name == "mt5_pcgrad":
            self.automatic_optimization = False
        # self.automatic_optimization = False
        # self.all_losses = []
        # self.label2id = config.label2id
        path = MODELS[model_name]["model_name"]
        child_module = MODELS[model_name]["module"]

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.peft = peft
        self.peft_checkpoint = peft_checkpoint
        self.config = AutoConfig.from_pretrained(path)
        self.config.label2id = label2id
        self.config.tasks = tasks
        self.config.num_accum_batches = num_accum_batches
        self.config.r4f = r4f
        self.config.r3f = r3f
        self.config.scale_loss = scale_loss
        self.config.retain_gradients = retain_grads

        self.tasks = dict(tasks)
        # child_model = child_module(self.config)
        child = child_module(self.config)
        self.child = self.wrap_with_peft(child) if peft_checkpoint else child

        self.batch_name_map_eval = batch_name_map_eval
        self.batch_name_map_test = batch_name_map_test
        self.eval_metrics = metrics.init_metrics(self.batch_name_map_eval)
        self.test_metrics = metrics.init_metrics(self.batch_name_map_test)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.predictions = {}

    def forward(self, input_ids, attention_mask=None, labels=None, *args, **kwargs):
        return self.child(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        if (
            self.peft
            and self.trainer.checkpoint_callback.current_score
            >= self.trainer.checkpoint_callback.best_model_score
            and (self.trainer.local_rank == 0 or self.trainer.local_rank == -1)
        ):
            self.child.model.save_pretrained(
                self.trainer.checkpoint_callback.dirpath + "/lora_ckpt"
            )

    def training_step(self, batch, batch_idx):
        loss = self.child.training_step(batch, batch_idx)

        # loss = self.scale_loss(loss, batch["task_name"], batch["task_size"])

        if loss is not None:
            self.log(
                "train_loss",
                loss,
                on_step=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, references = self.child.validation_step(
            batch, batch_idx, dataloader_idx
        )
        self.validation_step_outputs.append(
            {
                "dataloader_idx": dataloader_idx,
                "predictions": predictions,
                "references": references,
                "guids": batch["guids"] if batch["task_name"] == "wpr" else [],
            }
        )

    def on_validation_epoch_end(self):
        self.custom_accumulated_log(
            self.validation_step_outputs, self.batch_name_map_eval, self.eval_metrics
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, references = self.child.test_step(batch, batch_idx, dataloader_idx)
        self.test_step_outputs.append(
            {
                "dataloader_idx": dataloader_idx,
                "predictions": predictions,
                "references": references,
                "guids": batch["guids"] if batch["task_name"] == "wpr" else [],
            }
        )

    def on_test_epoch_end(self):
        self.custom_accumulated_log(
            self.test_step_outputs,
            self.batch_name_map_test,
            self.test_metrics,
            name_split="test/",
        )
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, references = self.child.predict_step(
            batch, batch_idx, dataloader_idx
        )
        if self.batch_name_map_test[dataloader_idx] not in self.predictions.keys():
            self.predictions[self.batch_name_map_test[dataloader_idx]] = {
                "predictions": [],
                "references": [],
                "guids": [],
            }
        self.predictions[self.batch_name_map_test[dataloader_idx]][
            "predictions"
        ] += predictions
        self.predictions[self.batch_name_map_test[dataloader_idx]][
            "references"
        ] += references
        self.predictions[self.batch_name_map_test[dataloader_idx]]["guids"] += (
            batch["guids"] if batch["task_name"] == "wpr" else []
        )

        return predictions

    def configure_optimizers(self):
        if self.peft:
            if not self.peft_checkpoint:
                self.child = self.wrap_with_peft(self.child)

            return torch.optim.Adam(self.parameters(), lr=5e-4)
        return self.child.configure_optimizers()

    def wrap_with_peft(self, child):
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        device = child.model.device
        child.model = get_peft_model(child.model, peft_config)
        child.model.to(device)
        child.model.print_trainable_parameters()
        return child

    def log_value(self, name, value, sync_dist=True):
        self.log(
            name,
            value,
            on_epoch=True,
            logger=True,
            sync_dist=sync_dist,
            add_dataloader_idx=False,
        )

    def custom_accumulated_log(
        self, step_outputs, batch_name_map, data_metrics, name_split="eval/"
    ):
        outs = {}
        for entry in step_outputs:
            key = entry["dataloader_idx"]
            if key not in outs.keys():
                outs[key] = {
                    "predictions": [],
                    "references": [],
                    "guids": [],
                }
            outs[key]["predictions"] += entry["predictions"]
            outs[key]["references"] += entry["references"]
            outs[key]["guids"] += entry["guids"]

        for dataloader_idx, data in outs.items():
            predictions = torch.tensor(data["predictions"])
            references = torch.tensor(data["references"])
            guids = torch.tensor(data["guids"])
            name = name_split + batch_name_map[dataloader_idx]
            if "wpr" in self.batch_name_map_eval[dataloader_idx]:
                data_metrics[dataloader_idx](
                    predictions.float(), references.long(), guids.cpu().long()
                )
                self.log_value(name + "_ncdg", data_metrics[dataloader_idx])
            elif "ctkfacts" in self.batch_name_map_eval[dataloader_idx]:
                data_metrics[dataloader_idx](predictions, references)
                self.log_value(name + "_f1", data_metrics[dataloader_idx])
            else:
                data_metrics[dataloader_idx](predictions, references)
                self.log_value(name + "_accuracy", data_metrics[dataloader_idx])

        avg = sum([metric.compute() for metric in data_metrics]) / len(batch_name_map)
        self.log_value(name_split + "accumulate", avg, sync_dist=False)


# %%
# from mt5 import T5

# path = "/home/semindan/baka/checkpoints/mt5_all_2/"
# #%%
# # #%%
# model = ModelModule.load_from_checkpoint("/home/semindan/baka/checkpoints/mt5_all_2/best.ckpt", model_name = "mt5", label2id=[], strict=False)
# # # %%
# # model.model.save_pretrained(path + "chk")
# # # %%
# # model.model.from_pretrained(path + "chk")
# # # %%
# #%%
# ckpt = torch.load(path + "best.ckpt", map_location=lambda storage, loc: storage)
# new_state_dict = OrderedDict()

# for k, v in ckpt["state_dict"].items():
#     if k[:6] != 'child.model':
#         name = "child." + k
#     else:
#         name = k
#     new_state_dict[name] = v

# model.load_state_dict(new_state_dict)

# tokenized = model.tokenizer("xnli: premise: aaa hypothesis: aaa", return_tensors="pt")
# model.tokenizer.batch_decode(model.child.model.generate(tokenized["input_ids"]))
# # %%
# # %%
# # %%
# model.save_checkpoint()
# # %%
