import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch.nn as nn
import torch.nn.functional as F
import transformers as tr
import torch
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pickle
from thesis.src.data.datamodule import DataModule
import argparse
import os
import random
from thesis.src.models.modelmodule import ModelModule


def init_callbacks(args):
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        # every_n_epochs=1,
        save_top_k=1,
        save_last=True,
        monitor="eval/accumulate",
        mode="max",
        dirpath=args.save_path + args.name,
        filename="best",
        save_on_train_epoch_end=False,
    )
    early_stop_callback = EarlyStopping(
        monitor="eval/accumulate",
        min_delta=0.0001,
        patience=35,
        verbose=False,
        mode="max",
        check_finite=True,
    )
    return [lr_monitor, checkpoint_callback, early_stop_callback]

def init_logger(args):
    wandb_logger = pl_loggers.WandbLogger(
        name=args.name,
        save_dir=args.save_path + args.name,
        project=args.wandb_project_name,
    )
    return wandb_logger

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.devices > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def set_precision():
    torch.set_float32_matmul_precision("medium")

def create_folder(args):
    if not os.path.exists(args.save_path + args.name):
        try:
            os.mkdir(args.save_path + args.name)
        except:
            pass

def get_devices(args):
    devices = list(range(int(args.devices)))
    return devices


def main(args):
    set_seed(args)
    set_precision()
    create_folder(args)
    callbacks = init_callbacks(args)
    wandb_logger = init_logger(args)
    devices = get_devices(args)
    
    dataset = DataModule(
        args.model,
        task_names=args.list_tasks,
        batch_size=args.batch_size,
        to_text=("mt5" in args.model),
        mix_mlm=args.mix,
        distributed=args.devices > 1,
        zero_shot_ctk=args.zero_shot_ctk,
        heterogenous_distributed=args.heterogenous,
        insert_prefix=args.prefix,
    )

    (
        batch_name_map_eval,
        batch_name_map_test,
        tasks,
        label2id_dict,
    ) = dataset.prepare_data()
    

    model = None
    if args.checkpoint:
        model = ModelModule.load_from_checkpoint(
            args.checkpoint,
            model_name = args.model,
            tasks = tasks,
            batch_name_map_eval = batch_name_map_eval,
            batch_name_map_test = batch_name_map_test,
            label2id = label2id_dict,
            peft=args.peft,
            peft_checkpoint = args.peft_checkpoint,
            num_accum_batches=args.accum_batches,
            r3f = args.r3f,
            r4f = args.r4f,
            scale_loss = args.scale_loss,
            strict = False)
    else:
        model = ModelModule(
            model_name = args.model,
            tasks = tasks,
            batch_name_map_eval = batch_name_map_eval,
            batch_name_map_test = batch_name_map_test,
            label2id = label2id_dict,
            peft_checkpoint = args.peft_checkpoint,
            peft=args.peft,
            num_accum_batches=args.accum_batches,
            r3f = args.r3f,
            r4f = args.r4f,
            scale_loss = args.scale_loss)

    wandb_logger.watch(model)

    trainer = pl.Trainer(
        strategy="ddp_find_unused_parameters_true" if "xlm" in args.model else "ddp",
        # strategy="ddp" if len(devices) > 1 else "auto",
        use_distributed_sampler=False,
        # fast_dev_run=5,self.trainer.checkpoint_callback.dirpath
        # limit_train_batches=2,
        # num_sanity_val_steps=5,
        # limit_val_batches=2,
        accelerator="auto",
        devices=devices,
        num_nodes=1,
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=args.val_every_n_steps * args.accum_batches,
        # val_check_interval=0.5,
        callbacks=callbacks,
        max_epochs=args.epochs,
        max_steps=args.steps,
        logger=wandb_logger,
        accumulate_grad_batches=1 if "pcgrad" in args.model else args.accum_batches,
    )

    if not args.eval_only:
        trainer.fit(
            model,
            datamodule=dataset,
        )


    ckpt_dir = args.save_path + args.name
    ckpt_path = ckpt_dir + "/" + "best.ckpt"
    trainer.validate(model, ckpt_path=ckpt_path, datamodule=dataset)
    trainer.test(model, ckpt_path=ckpt_path, datamodule=dataset)
    trainer.predict(model, ckpt_path=ckpt_path, datamodule=dataset)
    with open(ckpt_dir + "/" + args.name + ".pkl", "wb") as f:
        pickle.dump(model.predictions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model")
    parser.add_argument("--name", type=str, help="exp_name")
    parser.add_argument("--save_path", type=str, help="path", default="/home/semindan/baka/checkpoints_clean/")
    parser.add_argument("--wandb_project_name", type=str, help="path", default="thesis")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--devices", type=int, help="number of devices", default=1)
    parser.add_argument("--seed", type=int, help="seed", default=42)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=10)
    parser.add_argument("--steps", type=int, help="number of steps", default=16000)
    parser.add_argument("--accum_batches", type=int, help="accum when t5", default=2)
    parser.add_argument("--batch_size", type=int, help="initial batch size", default=16)
    parser.add_argument("--val_every_n_steps", type=int, help="frequency of eval", default=200)
    parser.add_argument(
        "--eval_only", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--peft", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--peft_checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefix", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--heterogenous", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--r3f", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--r4f", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scale_loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--zero_shot_ctk", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "-l",
        "--list_tasks",
        nargs="+",
        help="<Required> Set flag",
        required=True,
        default=None,
    )
    args = parser.parse_args()
    main(args)


