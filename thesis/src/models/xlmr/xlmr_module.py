import pytorch_lightning as pl
from thesis.src.models.xlmr.xlmr_model import XLMRobertaMultiTask
import torch
from transformers import get_constant_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from thesis.src.models.xlmr.xlmr_model import XLMRobertaMultiTask
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class XLMRModule(pl.LightningModule):
    def __init__(self, config, path="xlm-roberta-base"):
        super().__init__()
        self.save_hyperparameters()
        self.label2id = config.label2id
        self.tasks = config.tasks
        self.all_losses = []
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = XLMRobertaMultiTask.from_pretrained(
            path, config=config, tasks=self.tasks
        )
        eps = 1e-5
        self.r3f_lambda = 0.01
        self.do_r3f = config.r3f
        self.do_scale = config.scale_loss
        self.noise_sampler = torch.distributions.uniform.Uniform(low=-eps, high=eps)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        task_name=None,
        *args,
        **kwargs
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            name=task_name,
        )

    def compute_loss(self, logits, labels):
        loss_fct = CrossEntropyLoss(reduction="mean", ignore_index=-100)
        return loss_fct(logits, labels.view(-1))

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        labels = batch["labels"].to(out.logits.device)
        out_loss = self.compute_loss(out.logits, labels)
        loss = self.r3f(batch) if self.do_r3f else out_loss
        if self.do_scale:
            loss = self.scale_loss(loss, batch["task_name"])
        return loss

    def r3f(self, batch):
        inputs_embeds = self.model.roberta.embeddings(input_ids=batch["input_ids"])
        out = self(
            attention_mask=batch["attention_mask"],
            inputs_embeds=inputs_embeds,
            task_name=batch["task_name"],
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
        noised_out = self(
            attention_mask=batch["attention_mask"],
            inputs_embeds=noised_embeds,
            task_name=batch["task_name"],
        )

        return noised_out.logits

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
        n_classes = len(self.label2id[task_name])
        factor = torch.tensor(n_classes).log()
        return loss / factor

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(**batch)
        predictions = torch.argmax(out.logits, dim=-1)
        references = batch["labels"]
        return predictions, references

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(**batch)
        predictions = torch.argmax(out.logits, dim=-1)
        references = batch["labels"]
        return predictions, references

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        out = self(**batch)
        predictions = torch.argmax(out.logits, dim=-1)
        references = batch["labels"]
        return predictions, references

    def configure_optimizers(self):
        print("⚡", "using XLM-R", "⚡")
        optimizer = torch.optim.Adam(self.parameters(), lr=7.5e-6)
        return optimizer


# %%
