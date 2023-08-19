from thesis.src.models.mt5.mt5_module import MT5
from thesis.src.models.xlmr.xlmr_module import XLMRModule
from thesis.src.models.mt5.mt5_module_pcgrad import MT5PCGrad
from thesis.src.models.mbert.mbert_module import MBERTModule


MODELS = {
    "xlm-r": {"model_name": "xlm-roberta-base", "module": XLMRModule},
    "mt5": {"model_name": "google/mt5-small", "module": MT5},
    "mt5_pcgrad": {"model_name": "google/mt5-base", "module": MT5PCGrad},
    "mbert": {"model_name": "bert-base-multilingual-cased", "module": MBERTModule},
}

XGLUE_TASKS = ["paws-x", "nc", "xnli", "qadsm", "wpr", "qam"]
AIC_TASKS = ["ctkfacts_nli", "csfever_nli"]
AIC_PREFIX = "ctu-aic/"
XNLI_LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]
