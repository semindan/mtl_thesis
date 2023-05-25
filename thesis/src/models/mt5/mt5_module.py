from typing import Any
import pytorch_lightning as pl
from transformers import AutoTokenizer
from thesis.src.models.mt5.mt5_model import MT5ForConditionalGeneration
import torch
from thesis.src.optim.pcgrad import PCGrad
from torch_optimizer import Adafactor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


class MT5(pl.LightningModule):
    def __init__(self, config, path="google/mt5-base"):
        super().__init__()
        self.save_hyperparameters()
        self.label2id = config.label2id
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = MT5ForConditionalGeneration.from_pretrained(path, config=config)
        self.do_r3f = config.r3f
        self.do_scale = config.scale_loss
        eps = 1e-5
        self.r3f_lambda = 0.01
        # self.noise_sampler = torch.distributions.normal.Normal(loc=0.0, scale=eps)
        self.noise_sampler = torch.distributions.uniform.Uniform(low=-eps, high=eps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        *args,
        **kwargs
    ):
        return self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def compute_loss(self, logits, labels):
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        labels = batch["labels"].to(out.logits.device)
        out_loss = self.compute_loss(out.logits, labels)
        loss = self.r3f(batch) if self.do_r3f else out_loss
        if self.do_scale:
            loss = self.scale_loss(loss, batch["task_name"])
        return loss

    def r3f(self, batch):
        inputs_embeds = self.model.shared(batch["input_ids"])
        out = self(
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            inputs_embeds=inputs_embeds,
        )
        logits = out.logits
        noised_logits = self.get_noised_logits(batch, inputs_embeds)

        symm_kl = self._get_symm_kl(noised_logits, logits)
        sample_size = batch["labels"].view(-1).numel()

        labels = batch["labels"].to(logits.device)
        loss = self.compute_loss(logits, labels)

        loss_new = loss + self.r3f_lambda * symm_kl * sample_size

        return loss_new

    def get_noised_logits(self, batch, inputs_embeds):
        noise = self.noise_sampler.sample(sample_shape=inputs_embeds.shape).to(
            inputs_embeds
        )

        noised_embeds = inputs_embeds.detach().clone() + noise
        out = self(
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            inputs_embeds=noised_embeds,
        )

        noised_logits = out.logits
        return noised_logits

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
                log_target=True,
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
                log_target=True,
            )
        ) / noised_logits.size(0)

    def scale_loss(self, loss, task_name):
        if task_name == "mc4":
            n_classes = self.model.config.vocab_size
        else:
            n_classes = len(self.label2id[task_name])

        factor = torch.tensor(n_classes).log()
        return loss / factor

    def generate_predictions(self, input_ids, task_name):
        predictions = self.model.generate(input_ids=input_ids)
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
        optimizer = Adafactor(
            self.parameters(),
            lr=lr,
            relative_step=False,
            warmup_init=False,
            scale_parameter=True,
        )
        return optimizer
