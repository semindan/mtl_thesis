import torch
from torch import nn
import math
from typing import List, Optional, Tuple, Union
from transformers import XLMRobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import spectral_norm

# from transformers import XLMRobertaModel, XLMRobertaConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import transformers as tr
from thesis.src.models.xlmr.xlmr_roberta import XLMRobertaModel


def create_class_head(config, num_classes):
    return XLMRobertaClassificationHead(config, num_classes)


class XLMRobertaMultiTask(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, tasks):
        super().__init__(config)

        config.retain_gradients = (
            config.retain_gradients
            if ("retain_gradients" in config.__dict__)
            else False
        )
        self.config = config
        # self.roberta = XLMRobertaModel()
        self.roberta = XLMRobertaModel(config, add_pooling_layer=True)

        self.retain_gradients = config.retain_gradients

        self.heads = nn.ParameterDict(
            {
                str(task): create_class_head(self.config, num_classes)
                for task, num_classes in tasks
            }
        )
        # Initialize weights and apply final processing
        self.post_init()

    def _get_model_outputs(self, key):
        if key == "multihead_output":
            # get list (layers) of multihead module outputs
            return [
                layer.attention.self.multihead_output
                for layer in self.roberta.encoder.layer
            ]
        elif key == "layer_output":
            # get list of encoder LayerNorm layer outputs
            return [layer.output.layer_output for layer in self.roberta.encoder.layer]
        elif key == "cls_output":
            # get the final output of the model
            # return self.pooler.cls_output
            return self.roberta.pooler.cls_output
            # return None
        else:
            raise ValueError("Key not found: %s" % (key))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        name="",
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[1]

        # if self.retain_gradients:
        #     self.first_token_output = sequence_output[:, 0, :]
        #     self.first_token_output.retain_grad()

        # out = list()
        # for i, n in enumerate(name):
        #     model_out = self.heads[self.tasks_list[int(n)]](sequence_output[None, i])
        #     model_out_2 = F.pad(model_out, (0, self.n_classes_max - n_classes[i]), "constant", -100.0)
        #     out.append(model_out_2[0])
        # logits = torch.stack(out).requires_grad_(True)

        logits = self.heads[name](sequence_output)
        loss = None
        self.config.problem_type = "single_label_classification"
        loss_fct = nn.CrossEntropyLoss()

        if labels is not None:
            loss = loss_fct(logits, labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)
        if "r4f" in config.__dict__ and config.r4f:
            self.out_proj = spectral_norm(self.out_proj)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = features.mean(dim=1)
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# %%
# model = XLMRobertaMultiTask.from_pretrained("xlm-roberta-base")
# %%
# model.heads["paws-x"].dense.bias
# %%
