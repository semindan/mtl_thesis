import pytorch_lightning as pl
from thesis.src.models.xlmr.xlmr_model import XLMRobertaMultiTask
import torch
from transformers import get_constant_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import torch.nn.functional as F

class MBERTModule(pl.LightningModule):
    def __init__(self, config, path="bert-base-multilingual-cased"):
        super().__init__()
        self.save_hyperparameters()
        self.label2id = config.label2id
        self.tasks = config.tasks
        self.all_losses = []
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path,  num_labels=self.tasks[0][1])
    def forward(
        self, input_ids = None, attention_mask = None, labels = None, inputs_embeds=None, task_name = None, token_type_ids= None, *args, **kwargs
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds = inputs_embeds,
            token_type_ids = token_type_ids
        )

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        return out.loss
        
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
        print("⚡", "using mBERT", "⚡")
        optimizer = torch.optim.Adam(self.parameters(), lr=7.5e-6)
        return optimizer


# %%
