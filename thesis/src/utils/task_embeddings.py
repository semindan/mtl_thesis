import torch
import numpy as np
import argparse
import random
from thesis.src.data.datamodule import DataModule
import numpy as np
import torch
from tqdm import tqdm, trange
import joblib
from thesis.src.models.modelmodule import ModelModule


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.devices > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_Fisher_mt5(args, model, input_mask, total_tokens):
    outputs = {}
    base_model = model.model
    for name, parameter in base_model.named_parameters():
        if parameter.requires_grad:
            score = parameter.grad if args.feature_type == "grads" else parameter
            if score is not None and name not in outputs:
                score = score**args.pow
                outputs[name] = score
    # activations
    for key in ["multihead_output", "layer_output"]:
        model_outputs = base_model._get_model_outputs(key=key)
        for i in range(base_model.config.num_hidden_layers):
            name = "encoder.layer.{}.{}".format(i, key)
            model_outputs_i = (
                model_outputs[i].grad
                if args.feature_type == "grads"
                else model_outputs[i]
            )
            if model_outputs_i is not None:
                score = torch.einsum(
                    "ijk,ij->ijk",  # this einsum replaces padded elements with zeroes
                    [
                        model_outputs_i,  # batch_size x max_seq_length x hidden_size
                        input_mask.float(),
                    ],
                )  # batch_size x max_seq_length
                if score is not None and name not in outputs:
                    score = score.sum(0).sum(0)
                    score = score**args.pow
                    # normalize
                    score = score / total_tokens
                    outputs[name] = score

            raise NotImplementedError
    for key in [
        "decoder_multihead_output",
        "decoder_layer_output",
        "decoder_cross_attention_output",
    ]:
        model_outputs = base_model._get_model_outputs(key=key)
        for i in range(base_model.config.num_hidden_layers):
            name = "decoder.layer.{}.{}".format(i, key)
            model_outputs_i = (
                model_outputs[i].grad
                if args.feature_type == "grads"
                else model_outputs[i]
            )

            if model_outputs_i is not None:
                score = model_outputs_i
                if score is not None and name not in outputs:
                    score = score.sum(0).sum(0)
                    score = score**args.pow
                    # normalize
                    score = score / total_tokens
                    outputs[name] = score
    return outputs


def compute_Fisher(args, model, input_mask, total_tokens):
    outputs = {}

    base_model = model.model
    for name, parameter in base_model.roberta.named_parameters():
        if parameter.requires_grad:
            score = parameter.grad if args.feature_type == "grads" else parameter
            if score is not None and name not in outputs:
                score = score**args.pow
                outputs[name] = score
    # activations
    for key in ["multihead_output", "layer_output"]:
        model_outputs = base_model._get_model_outputs(key=key)
        for i in range(base_model.config.num_hidden_layers):
            name = "encoder.layer.{}.{}".format(i, key)
            # print(name, key)
            # print(model_outputs[i])
            model_outputs_i = (
                model_outputs[i].grad
                if args.feature_type == "grads"
                else model_outputs[i]
            )

            if model_outputs_i is not None:
                score = torch.einsum(
                    "ijk,ij->ijk",
                    [
                        model_outputs_i,  # batch_size x max_seq_length x hidden_size
                        input_mask.float(),
                    ],
                )  # batch_size x max_seq_length
                if score is not None and name not in outputs:
                    score = score.sum(0).sum(0)
                    score = score**args.pow
                    # normalize
                    score = score / total_tokens
                    outputs[name] = score
    # cls output
    name = "cls_output"
    score = (
        base_model._get_model_outputs(key=name).grad
        if args.feature_type == "grads"
        else base_model._get_model_outputs(key=name)
    )  # batch_size x hidden_size

    if score is not None and name not in outputs:
        score = score.sum(0)
        score = score**args.pow
        # normalize
        score = score / total_tokens
        outputs[name] = score

    # task-specific layer
    for name, parameter in model.named_parameters():
        if "roberta" not in name:
            score = parameter.grad if args.feature_type == "grads" else parameter
            if score is not None and name not in outputs:
                score = score**args.pow
                outputs[name] = score

    # print(outputs.keys())
    return outputs


def compute_Fisher_no_labels(args, model, input_mask, logits):
    total_tokens = input_mask.float().detach().sum().data
    #  We are doing classification
    softmax_logits = torch.softmax(logits, dim=1)  # batch_size x num_labels
    sampled_indices = torch.multinomial(softmax_logits, args.num_trials_for_FIM, True)
    log_softmax_logits = torch.log(softmax_logits)
    sampled_log_softmax_logits = torch.gather(
        log_softmax_logits, dim=1, index=sampled_indices
    )

    sampled_log_softmax_logits = (
        sampled_log_softmax_logits.sum(0).sum(0) / sampled_log_softmax_logits.numel()
    )

    model.zero_grad()
    sampled_log_softmax_logits.backward()
    outputs = compute_Fisher(args, model, input_mask, total_tokens)
    return outputs


def compute_Fisher_with_labels(args, model, input_mask, loss):
    total_tokens = input_mask.float().detach().sum().data

    model.zero_grad()
    loss.backward()
    if "xlm" in args.model:
        outputs = compute_Fisher(args, model, input_mask, total_tokens)
    else:
        outputs = compute_Fisher_mt5(args, model, input_mask, total_tokens)
    return outputs


def compute_taskemb(args, task_name, train_dataloader, model):
    # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    total_num_examples = 0
    model.zero_grad()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    global_feature_dict = {}
    total_num_examples = 0
    for batch in tqdm(train_dataloader):
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        loss, logits = outputs[0], outputs[1]
        input_mask = batch["attention_mask"]

        if not args.use_labels:
            feature_dict = compute_Fisher_no_labels(args, model, input_mask, logits)
        else:
            feature_dict = compute_Fisher_with_labels(args, model, input_mask, loss)
        ###
        if len(global_feature_dict) == 0:
            for key in feature_dict:
                global_feature_dict[key] = feature_dict[key].detach().cpu().numpy()
        else:
            for key in feature_dict:
                global_feature_dict[key] += feature_dict[key].detach().cpu().numpy()

        model.zero_grad()
        total_num_examples += batch["input_ids"].size(0)

    # Normalize
    for key in global_feature_dict:
        global_feature_dict[key] = global_feature_dict[key] / total_num_examples
    print("Done!")
    # Save features
    with open(
        "embeddings/" + args.model + "_" + task_name + "_taskemb" + ".pkl", "wb"
    ) as f:
        joblib.dump(global_feature_dict, f)
        # pickle.dump(global_feature_dict, f)


def compute_textemb(args, task_name, train_dataloader, model):
    device = torch.device("cuda")
    vec_dict = {}
    total_examples = 0

    model.to(device)
    for batch in tqdm(train_dataloader):
        model.eval()
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None,
            )
            sequence_output = outputs[0]
            input_mask = batch["attention_mask"]
            active_sequence_output = torch.einsum(
                "abc,ab -> abc", [sequence_output, input_mask]
            )
            avg_sequence_output = active_sequence_output.sum(1) / input_mask.sum(
                dim=1
            ).view(input_mask.size(0), 1)
            sum_avg_seq_out = avg_sequence_output.sum(dim=0).detach().cpu().numpy()
            if not vec_dict:
                vec_dict["avg_feature_vec"] = sum_avg_seq_out
            else:
                vec_dict["avg_feature_vec"] += sum_avg_seq_out
        total_examples += input_mask.size(0)

    vec_dict["avg_feature_vec"] = vec_dict["avg_feature_vec"] / total_examples
    with open(
        "embeddings/" + args.model + "_" + task_name + "_textemb" + ".pkl", "wb"
    ) as f:
        # pickle.dump(vec_dict, f)
        joblib.dump(vec_dict, f)


def compute_textemb_mt5(args, task_name, train_dataloader, model):
    device = torch.device("cuda")
    vec_dict = {}
    total_examples = 0

    model.to(device)
    for batch in tqdm(train_dataloader):
        model.eval()
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_attentions=None,
                output_hidden_states=True,
                return_dict=None,
            )
            sequence_output = outputs[0]
            input_mask = batch["attention_mask"]
            active_sequence_output = torch.einsum(
                "abc,ab -> abc", [sequence_output, input_mask]
            )
            avg_sequence_output = active_sequence_output.sum(1) / input_mask.sum(
                dim=1
            ).view(input_mask.size(0), 1)
            sum_avg_seq_out = avg_sequence_output.sum(dim=0).detach().cpu().numpy()
            if not vec_dict:
                vec_dict["avg_feature_vec"] = sum_avg_seq_out
            else:
                vec_dict["avg_feature_vec"] += sum_avg_seq_out
        total_examples += input_mask.size(0)

    vec_dict["avg_feature_vec"] = vec_dict["avg_feature_vec"] / total_examples
    with open(
        "embeddings/" + args.model + "_" + task_name + "_textemb" + ".pkl", "wb"
    ) as f:
        # pickle.dump(vec_dict, f)
        joblib.dump(vec_dict, f)


# %%
def main(args):
    set_seed(args)
    retain_grads = args.embed_type == "taskemb"
    task = args.list_tasks[0]
    dataset = DataModule(
        args.model,
        task_names=args.list_tasks,
        batch_size=args.batch_size,
        to_text=args.model == "mt5",
        insert_prefix=True,
    )

    (
        batch_name_map_eval,
        batch_name_map_test,
        tasks,
        label2id_dict,
    ) = dataset.prepare_data()
    dataset.setup("fit")

    model = None
    if args.checkpoint:
        model = ModelModule.load_from_checkpoint(
            args.checkpoint,
            model_name=args.model,
            tasks=tasks,
            batch_name_map_eval=batch_name_map_eval,
            batch_name_map_test=batch_name_map_test,
            retain_grads=retain_grads,
        )
    else:
        model = ModelModule(
            args.model,
            tasks,
            batch_name_map_eval,
            batch_name_map_test,
            label2id=label2id_dict,
            retain_grads=retain_grads,
        )

    print("loaded")
    if args.embed_type == "textemb":
        print("textemb start")
        if args.model == "mt5":
            compute_textemb_mt5(
                args, task, dataset.train_dataloader(), model.child.model.encoder
            )
        else:
            compute_textemb(
                args, task, dataset.train_dataloader(), model.child.model.roberta
            )
    elif args.embed_type == "taskemb":
        print("taskemb start")
        compute_taskemb(args, task, dataset.train_dataloader(), model.child)
    else:
        raise ValueError("no command")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embed_type", type=str, default="textemb", help="taskemb or textemb"
    )
    parser.add_argument("--devices", type=int, help="number of devices", default=1)
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument(
        "-l",
        "--list_tasks",
        nargs="+",
        help="<Required> Set flag",
        required=True,
        default=None,
    )

    parser.add_argument(
        "--pow", type=float, default=2.0, help="Return features to the power pow."
    )
    parser.add_argument(
        "--feature_type",
        default="grads",
        type=str,
        help="The type of the features selected in ['grads', 'weights']",
    )
    parser.add_argument(
        "--retain_gradients",
        default=True,
        type=eval,
        help="Whether to retain gradients at each layer output of the feature extractor.",
    )
    parser.add_argument(
        "--do_pooling",
        default=True,
        type=eval,
        help="Whether to pool the feature extractor.",
    )
    parser.add_argument(
        "--use_labels",
        default=True,
        type=eval,
        help="Whether to use training labels or sample from the model's predictive distribution \n"
        "pθ(y|xn), e.g., to compute the theoretical Fisher information.",
    )
    parser.add_argument(
        "--num_trials_for_FIM",
        type=int,
        default=100,
        help="Number of trials to sample from the model's predictive distribution pθ(y|xn).",
    )
    parser.add_argument(
        "--FIM_scale",
        type=float,
        default=0.25,
        help="Standard deviation of the distribution used to compute the theoretical FIM.",
    )
    parser.add_argument(
        "--finetune_classifier",
        default=False,
        type=eval,
        help="Whether to fine-tune the final classifier.",
    )
    parser.add_argument(
        "--finetune_feature_extractor",
        default=False,
        type=eval,
        help="Whether to fine-tune the feature extractor.",
    )
    parser.add_argument("--model", type=str, help="model", default="mt5")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, help="initial batch size", default=4)

    args = parser.parse_args()

    main(args)
