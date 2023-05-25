from sklearn.metrics import ndcg_score
from evaluate import load
import evaluate
import numpy as np
import torchmetrics
import torch.nn as nn


def init_metrics(names):
    metrics = []
    for name in names:
        if "wpr" in name:
            metrics.append(torchmetrics.RetrievalNormalizedDCG())
        elif "ctk" in name:
            metrics.append(torchmetrics.F1Score(task="multiclass", num_classes=3))
        else:
            metrics.append(torchmetrics.Accuracy(task="multiclass", num_classes=10))
    metrics = nn.ModuleList(metrics)
    return metrics


def simple_ndcg(preds, labels, guids):
    ndcgs = []
    query2content = {}
    for guid, pred, label in zip(guids, preds, labels):
        query = guid
        if not query in query2content:
            query2content[query] = [[float(pred)], [float(label)]]
        else:
            query2content[query][0].append(float(pred))
            query2content[query][1].append(float(label))

    for key in query2content.keys():
        if len(query2content[key][1]) < 2 or len(query2content[key][0]) < 2:
            continue
        ndcgs.append(
            ndcg_score(
                np.asarray([query2content[key][1]]), np.asarray([query2content[key][0]])
            )
        )
    print(ndcgs)
    return np.array(ndcgs).mean()


def compute_metrics(eval_preds):
    logits, (labels, n_labels), inputs = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    results = metric.compute(predictions=predictions, references=labels)
    return results


def get_guids(data, col_name="query"):
    guids = []
    guid = 0
    last_query = ""
    for item in data:
        if (not item[col_name] == last_query) and len(last_query) > 0:
            guid += 1
        last_query = item[col_name]
        guids.append(guid)
    return guids
