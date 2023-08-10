from typing import Any
import lightning.pytorch as pl
from transformers import AutoTokenizer
from thesis.src.models.mt5.mt5_model import MT5ForConditionalGeneration
import torch
from thesis.src.optim.pcgrad import PCGrad
from torch_optimizer import Adafactor


class MT5PCGrad(pl.LightningModule):
    def __init__(self, config, path="google/mt5-base"):
        super().__init__()
        self.save_hyperparameters()
        self.label2id = config.label2id
        self.num_accum_batches = (
            2
            if "num_accum_batches" not in config.__dict__.keys()
            else config.num_accum_batches
        )
        self.automatic_optimization = False
        self.all_losses = []
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = MT5ForConditionalGeneration.from_pretrained(path, config=config)

    def forward(self, input_ids, attention_mask=None, labels=None, *args, **kwargs):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        y_hat = self(**batch)
        loss = y_hat.loss
        self.trainer.strategy.broadcast(loss, src=0)
        self.all_losses += list(self.trainer.strategy.all_gather(loss, sync_grads=True))

        if (batch_idx + 1) % self.num_accum_batches != 0 and batch_idx != 0:
            return None

        objectives = self.all_losses
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            opt._optim.zero_grad(set_to_none=True)
            self.manual_backward(obj, retain_graph=True)
            grad, shape, has_grad = opt._retrieve_grad()
            grads.append(opt._flatten_grad(grad, shape))
            has_grads.append(opt._flatten_grad(has_grad, shape))
            shapes.append(shape)

        pc_grad = opt._project_conflicting(grads, has_grads)
        pc_grad = opt._unflatten_grad(pc_grad, shapes[0])
        opt._set_grad(pc_grad)
        opt.step()

        self.all_losses.clear()
        return loss

    def generate_predictions(self, input_ids, task_name):
        predictions = self.model.generate(input_ids)
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        predictions = torch.tensor(
            [
                -1.0
                if word not in self.label2id[task_name]
                else self.label2id[task_name][word]
                for word in predictions
            ]
        )
        return predictions

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.generate_predictions(batch["input_ids"], batch["task_name"])
        references = batch["label"]
        return predictions, references

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.generate_predictions(batch["input_ids"], batch["task_name"])
        references = batch["label"]
        return predictions, references

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        predictions = self.generate_predictions(batch["input_ids"], batch["task_name"])
        references = batch["label"]
        return predictions, references

    def configure_optimizers(self):
        print("⚡", "using T5", "⚡")
        lr = 1e-3
        optimizer_inner = Adafactor(
            self.parameters(),
            lr=lr,
            relative_step=False,
            warmup_init=False,
            scale_parameter=True,
        )
        optimizer = PCGrad(self.parameters(), optimizer_inner, lr=lr)
        return optimizer
