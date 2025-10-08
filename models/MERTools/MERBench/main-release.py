import argparse
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from toolkit.dataloader import get_dataloaders
from toolkit.globals import runtime_config
from toolkit.models import get_models
from toolkit.utils.functions import func_random_select, func_update_storage, merge_args_config
from toolkit.utils.loss import CELoss, MSELoss
from toolkit.utils.metric import (
    average_folder_for_emos,
    average_folder_for_vals,
    gain_cv_results,
    gain_metric_from_results,
)
from torch import optim

# Constants for magic values
BIMODAL_FEATURES = 2
TRIMODAL_FEATURES = 3


@dataclass
class ModelContext:
    model: Any
    reg_loss: Any
    cls_loss: Any
    optimizer: torch.optim.Optimizer | None = None


def _move_to_cuda(batch, emos, vals):
    for key in batch:
        batch[key] = batch[key].cuda()
    return batch, emos.cuda(), vals.cuda()


def _forward_and_loss(args, context, batch, emos, vals):
    _, emos_out, vals_out, interloss = context.model(batch)
    loss = interloss
    emo_probs, emo_labels, val_preds, val_labels = None, None, None, None

    if args.output_dim1 != 0:
        loss = loss + context.cls_loss(emos_out, emos)
        emo_probs = emos_out.data.cpu().numpy()
        emo_labels = emos.data.cpu().numpy()
    if args.output_dim2 != 0:
        loss = loss + context.reg_loss(vals_out, vals)
        val_preds = vals_out.data.cpu().numpy()
        val_labels = vals.data.cpu().numpy()
    return loss, emo_probs, emo_labels, val_preds, val_labels


def _optimize(context, loss):
    assert context.optimizer is not None
    context.optimizer.zero_grad()
    loss.backward()
    if hasattr(context.model, "model") and getattr(context.model.model, "grad_clip", -1) != -1:
        torch.nn.utils.clip_grad_value_(
            [param for param in context.model.parameters() if param.requires_grad],
            context.model.model.grad_clip,
        )
    context.optimizer.step()


def train_or_eval_model(
    args,
    context: ModelContext,
    dataloader,
    epoch,
    train: bool = False,
):
    vidnames = []
    val_preds, val_labels = [], []
    emo_probs, emo_labels = [], []
    losses = []

    assert not train or context.optimizer is not None
    runtime_config.train = train
    context.model.train() if train else context.model.eval()

    for i, data in enumerate(dataloader):
        batch, emos, vals, bnames = data
        vidnames += bnames
        batch, emos, vals = _move_to_cuda(batch, emos, vals)

        loss, emo_prob, emo_label, val_pred, val_label = _forward_and_loss(
            args, context, batch, emos, vals
        )

        if emo_prob is not None:
            emo_probs.append(emo_prob)
            emo_labels.append(emo_label)
        if val_pred is not None:
            val_preds.append(val_pred)
            val_labels.append(val_label)
        losses.append(loss.data.cpu().numpy())

        if train:
            _optimize(context, loss)

        if (i + 1) % args.print_iters == 0:
            print(f"process on {i + 1}|{len(dataloader)}, meanloss: {np.mean(losses)}")

        if train and args.savemodel and (i + 1) % args.save_iters == 0:
            save_path = f"{save_modelroot}/{prefix_name}_epoch:{epoch:02d}_iter:{i + 1:06d}_meanloss:{np.mean(losses)}_{name_time}"
            context.model.model.save_pretrained(save_path)

    if emo_probs:
        emo_probs = np.concatenate(emo_probs)
    if emo_labels:
        emo_labels = np.concatenate(emo_labels)
    if val_preds:
        val_preds = np.concatenate(val_preds)
    if val_labels:
        val_labels = np.concatenate(val_labels)
    results, _ = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
    return dict(
        names=vidnames,
        loss=np.mean(losses),
        **results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Params for datasets
    parser.add_argument("--dataset", type=str, default=None, help="dataset")
    parser.add_argument(
        "--train_dataset", type=str, default=None, help="train dataset"
    )  # for cross-corpus test
    parser.add_argument(
        "--test_dataset", type=str, default=None, help="test dataset"
    )  # for cross-corpus test
    parser.add_argument(
        "--save_root", type=str, default="./saved", help="save prediction results and models"
    )
    parser.add_argument(
        "--debug", action="store_true", default=False, help="whether use debug to limit samples"
    )
    parser.add_argument(
        "--savemodel",
        action="store_true",
        default=False,
        help="whether to save model, default: False",
    )
    parser.add_argument("--save_iters", type=int, default=1e8, help="save models per iters")

    # Params for feature inputs
    parser.add_argument("--audio_feature", type=str, default=None, help="audio feature name")
    parser.add_argument("--text_feature", type=str, default=None, help="text feature name")
    parser.add_argument("--video_feature", type=str, default=None, help="video feature name")
    parser.add_argument(
        "--feat_type", type=str, default=None, help="feature type [utt, frm_align, frm_unalign]"
    )
    parser.add_argument(
        "--feat_scale",
        type=int,
        default=None,
        help="pre-compress input from [seqlen, dim] -> [seqlen/scale, dim]",
    )
    # Params for raw inputs
    parser.add_argument("--e2e_name", type=str, default=None, help="e2e pretrained model names")
    parser.add_argument(
        "--e2e_dim", type=int, default=None, help="e2e pretrained model hidden size"
    )

    # Params for model
    parser.add_argument(
        "--n_classes", type=int, default=None, help="number of classes [defined by args.label_path]"
    )
    parser.add_argument(
        "--hyper_path",
        type=str,
        default=None,
        help="whether choose fixed hyper-params [default use hyperparam tuning]",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="model name for training [mlp, attention, and others from MMSA toolkits]",
    )

    # Params for training
    parser.add_argument(
        "--lr", type=float, default=None, metavar="lr", help="set lr rate"
    )  # 如果是None, lr 作为了 hyper-params 通过 model-tune.yaml 调节
    parser.add_argument(
        "--lr_adjust",
        type=str,
        default="case1",
        help="[case1, case2]. case1: uniform lr; case2: pretrain lr = 1/10 fc lr",
    )
    parser.add_argument(
        "--l2", type=float, default=0.00001, metavar="L2", help="L2 regularization weight"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, metavar="BS", help="batch size [deal with OOM]"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, metavar="nw", help="number of workers"
    )
    parser.add_argument("--epochs", type=int, default=100, metavar="E", help="number of epochs")
    parser.add_argument("--print_iters", type=int, default=1e8, help="print per-iteartion")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    print("====== Params Pre-analysis =======")
    # accelate: pre-compress for ['frm_align', 'frm_unalign']
    if args.feat_type == "utt":
        args.feat_scale = 1
    elif args.feat_type == "frm_align":
        assert args.audio_feature.endswith("FRA")
        assert args.text_feature.endswith("FRA")
        assert args.video_feature.endswith("FRA")
        args.feat_scale = 6
    elif args.feat_type == "frm_unalign":
        assert args.audio_feature.endswith("FRA")
        assert args.text_feature.endswith("FRA")
        assert args.video_feature.endswith("FRA")
        args.feat_scale = 12

    # define store folder
    if args.train_dataset is not None:
        args.save_root = f"{args.save_root}-cross"
    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    whole_features = [item for item in whole_features if item is not None]
    if len(set(whole_features)) == 0:
        args.save_root = f"{args.save_root}-others"
    elif len(set(whole_features)) == 1:
        args.save_root = f"{args.save_root}-unimodal"
    elif len(set(whole_features)) == BIMODAL_FEATURES:
        args.save_root = f"{args.save_root}-bimodal"
    elif len(set(whole_features)) == TRIMODAL_FEATURES:
        args.save_root = f"{args.save_root}-trimodal"

    # generate model_config
    if args.hyper_path is None:
        model_config = OmegaConf.load("toolkit/model-tune.yaml")[args.model]
        model_config = func_random_select(model_config)
    else:
        model_config = OmegaConf.load(args.hyper_path)[args.model]
    runtime_config.dataset = args.dataset
    args = merge_args_config(args, model_config)  # merge params
    print("args: ", args)

    # save root
    save_resroot = os.path.join(args.save_root, "result")
    save_modelroot = os.path.join(args.save_root, "model")
    if not os.path.exists(save_resroot):
        os.makedirs(save_resroot)
    if not os.path.exists(save_modelroot):
        os.makedirs(save_modelroot)
    # gain prefix_name
    feature_name = "+".join(sorted(set(whole_features)))  # sort to avoid random order
    model_name = f"{args.model}+{args.feat_type}+{args.e2e_name}"
    prefix_name = f"features:{feature_name}_dataset:{args.dataset}_model:{model_name}"
    if args.train_dataset is not None:
        assert args.test_dataset is not None
        prefix_name += f"_train:{args.train_dataset}_test:{args.test_dataset}"

    print("====== Reading Data =======")
    dataloader_class = get_dataloaders(args)  # (MER2023 + e2e + e2e_name)
    train_loaders, eval_loaders, test_loaders = dataloader_class.get_loaders()
    assert len(train_loaders) == len(eval_loaders)
    print(f"train&val folder:{len(train_loaders)}; test sets:{len(test_loaders)}")
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()

    print("====== Training and Evaluation =======")
    folder_save = []  # store best results for each folder
    folder_duration = []
    for ii in range(len(train_loaders)):
        print(f">>>>> Cross-validation: training on the {ii + 1} folder >>>>>")
        train_loader = train_loaders[ii]
        eval_loader = eval_loaders[ii]
        start_time = name_time = time.time()

        print("Step1: build model (each folder has its own model)")
        model = get_models(args).cuda()
        reg_loss = MSELoss().cuda()
        cls_loss = CELoss().cuda()

        if args.lr_adjust == "case1":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif args.lr_adjust == "case2":
            assert args.model == "e2e_model", "lr_adjust=case2 only support for e2e_model"
            print("set different learning rates for different layers")
            optimizer = optim.Adam(
                [
                    {"params": model.model.pretrain_model.parameters(), "lr": args.lr / 10},
                    {"params": model.model.encoder.parameters(), "lr": args.lr},
                    {"params": model.model.fc_out_1.parameters(), "lr": args.lr},
                    {"params": model.model.fc_out_2.parameters(), "lr": args.lr},
                ],
                lr=args.lr,
                weight_decay=args.l2,
            )

        context = ModelContext(
            model=model, reg_loss=reg_loss, cls_loss=cls_loss, optimizer=optimizer
        )

        print("Step2: training (multiple epoches)")
        whole_store = []
        whole_metrics = []
        for epoch in range(args.epochs):
            epoch_store: dict[str, object] = {}

            # training and validation
            train_results = train_or_eval_model(args, context, train_loader, epoch, train=True)
            eval_context = ModelContext(
                model=model, reg_loss=reg_loss, cls_loss=cls_loss, optimizer=None
            )
            eval_results = train_or_eval_model(args, eval_context, eval_loader, epoch, train=False)
            func_update_storage(inputs=eval_results, prefix="eval", outputs=epoch_store)

            # use args.metric_name to determine best_index
            train_metric = gain_metric_from_results(train_results, args.metric_name)
            eval_metric = gain_metric_from_results(eval_results, args.metric_name)
            whole_metrics.append(eval_metric)
            print(
                f"epoch:{epoch + 1}; metric:{args.metric_name}; train results:{train_metric:.4f}; eval results:{eval_metric:.4f}"
            )

            # testing and saving
            for jj, test_loader in enumerate(test_loaders):
                test_context = ModelContext(
                    model=model, reg_loss=reg_loss, cls_loss=cls_loss, optimizer=None
                )
                test_results = train_or_eval_model(
                    args, test_context, test_loader, epoch, train=False
                )
                func_update_storage(
                    inputs=test_results, prefix=f"test{jj + 1}", outputs=epoch_store
                )

            # saving
            whole_store.append(epoch_store)

        print(f"Step3: saving and testing on the {ii + 1} folder")
        best_index = np.argmax(np.array(whole_metrics))
        folder_save.append(whole_store[best_index])
        end_time = time.time()
        duration = end_time - start_time
        folder_duration.append(duration)
        print(
            f">>>>> Finish: training on the {ii + 1}-th folder, best_index: {best_index}, duration: {duration} >>>>>"
        )
        # clear memory
        del model
        if "optimizer" in locals():
            del optimizer
        torch.cuda.empty_cache()

    print("====== Prediction and Saving =======")
    args.duration = np.sum(folder_duration)  # store duration
    cv_result = gain_cv_results(folder_save)
    save_path = f"{save_resroot}/cv_{prefix_name}_{cv_result}_{name_time}.npz"
    print(f"save results in {save_path}")
    np.savez_compressed(save_path, args=np.array(args, dtype=object))

    # store test1|test2|test3 results
    for jj in range(len(test_loaders)):
        emo_labels, emo_probs = average_folder_for_emos(folder_save, f"test{jj + 1}")
        val_labels, val_preds = average_folder_for_vals(folder_save, f"test{jj + 1}")
        _, test_result = dataloader_class.calculate_results(
            emo_probs, emo_labels, val_preds, val_labels
        )
        save_path = f"{save_resroot}/test{jj + 1}_{prefix_name}_{test_result}_{name_time}.npz"
        print(f"save results in {save_path}")
        np.savez_compressed(save_path, args=np.array(args, dtype=object))
