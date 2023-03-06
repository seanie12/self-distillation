import os

import numpy as np
import timm.optim.optim_factory as optim_factory
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (ViTConfig, ViTForImageClassification, ViTMAEConfig,
                          ViTMAEForPreTraining, ViTModel)
from transformers.optimization import get_linear_schedule_with_warmup

from dataloader import get_dataloader
from utils import Augmentation, InfIterator, accuracy


class BaseTrainer(object):
    def __init__(self) -> None:
        pass

    def train(self):
        pass

    def save(self):
        pass


def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    all_loss = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="evaluation", position=1, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)[0]
            loss = criterion(logits, y)
            all_loss.append(loss.item())

            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(y).sum()
            total += y.size(0)
    acc = 1.0 * correct.item() / total
    test_loss = np.mean(all_loss)

    return acc, test_loss


class PreTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        self.args = args

    def train(self):
        args = self.args
        device = torch.cuda.current_device()
        config = ViTMAEConfig.from_pretrained(args.model_name)
        config.update(
            {
                "mask_ratio": args.mask_ratio,  # 0.75
                "norm_pix_loss": args.norm_pix_loss,  # True
            }
        )
        # print(config)
        model = ViTMAEForPreTraining.from_pretrained(
            args.model_name, config=config)
        args.meta_algo = "further-pretraining"
        model = model.to(device)

        # optimizer
        param_groups = optim_factory.add_weight_decay(
            model, args.weight_decay)
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.95))

        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_steps, args.steps)
        data_loader, _, _, _ = get_dataloader(args.dataset, args.batch_size, args.workers,
                                              train_val_split=1.0, model_name=args.model_name)

        train_iter = InfIterator(data_loader)

        with tqdm(total=args.steps, leave=False, desc="pretraining") as pbar:
            for global_step in range(args.steps):
                batch = next(train_iter)
                samples, _ = batch
                samples = samples.to(device)

                model.train()
                outputs = model(samples)
                loss = outputs[0]

                model.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

                pbar.set_postfix({"Epoch": f"{global_step}/{args.steps}",
                                  "loss": "{0:.4f}".format(loss.item())})
                pbar.update(1)

        ckpt_folder = os.path.join(
            "./checkpoints", f"further-pretrain-{args.steps}", args.dataset)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        ckpt_file = os.path.join(ckpt_folder, "model.pt")

        ckpt = {"args": args,
                "state_dict": model.vit.state_dict()}
        torch.save(ckpt, ckpt_file)


class SelfPreTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        self.args = args

    def get_student_features(self, model, pixel_values):
        # get features without masking
        patch_embeddings = model.get_input_embeddings()
        position_embeddings = model.vit.embeddings.position_embeddings
        cls_token = model.vit.embeddings.cls_token

        embeddings = patch_embeddings(pixel_values)
        embeddings = embeddings + position_embeddings[:, 1:, :]

        # append cls token
        cls_token = cls_token + position_embeddings[:, :1, :]
        cls_token = cls_token.expand(embeddings.size(0), -1, -1)
        embeddings = torch.cat([cls_token, embeddings], dim=1)

        head_mask = model.get_head_mask(None, model.config.num_hidden_layers)

        encoder_outputs = model.vit.encoder(
            embeddings,
            head_mask=head_mask,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = model.vit.layernorm(sequence_output)

        return sequence_output[:, 0, :]

    def train(self):
        args = self.args
        device = torch.cuda.current_device()
        config = ViTMAEConfig.from_pretrained(args.model_name)
        config.update(
            {
                "mask_ratio": args.mask_ratio,  # 0.75
                "norm_pix_loss": args.norm_pix_loss,  # True
            }
        )
        print(config)
        model = ViTMAEForPreTraining.from_pretrained(
            args.model_name, config=config)
        teacher = ViTModel.from_pretrained(
            args.model_name, add_pooling_layer=False)

        # load teacher
        teacher_folder = os.path.join(
            "checkpoints", f"further-pretrain-{args.steps}", args.dataset)
        ckpt_file = os.path.join(teacher_folder, "model.pt")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        teacher.load_state_dict(ckpt["state_dict"])

        if torch.cuda.device_count() > 1:
            teacher_device = device + 1
        else:
            teacher_device = device
        teacher = teacher.to(teacher_device)

        model = model.to(device)
        # optimizer
        param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.95))

        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_steps, args.steps)
        data_loader, _, _, _ = get_dataloader(args.dataset, args.batch_size, args.workers,
                                              train_val_split=1.0, model_name=args.model_name)

        train_iter = InfIterator(data_loader)
        mse_criterion = nn.MSELoss(reduction="sum")
        model.train()
        teacher.eval()
        with tqdm(total=args.steps, leave=False, desc="pretraining") as pbar:
            for global_step in range(args.steps):
                batch = next(train_iter)
                samples, _ = batch
                samples = samples.to(device)

                # Masked Auto-Encoding loss
                outputs = model(samples)
                loss = outputs[0]

                model.zero_grad()
                loss.backward()

                with torch.no_grad():
                    teacher_h = teacher(samples.to(teacher_device))[0]
                    teacher_h = teacher_h[:, 0, :].to(device)

                student_h = self.get_student_features(model, samples)
                batch_size = samples.size(0)

                zero_matching = mse_criterion(student_h, teacher_h.detach())

                distill_loss = zero_matching / batch_size
                distill_loss.backward()

                optimizer.step()
                scheduler.step()

                pbar.set_postfix({"Epoch": f"{global_step}/{args.steps}",
                                  "loss": "{0:.4f}".format(loss.item())})
                pbar.update(1)

        ckpt_folder = os.path.join(
            "./checkpoints", f"self-further-pretrain-{args.steps}", args.dataset)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        ckpt_file = os.path.join(ckpt_folder, "model.pt")

        ckpt = {"args": args, "state_dict": model.vit.state_dict()}
        torch.save(ckpt, ckpt_file)


class FinetuningTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def train(self):
        args = self.args
        train_loader, _, test_loader, num_classes = get_dataloader(args.dataset, args.batch_size, args.workers,
                                                                   train_val_split=1.0, model_name=args.model_name)

        train_iter = InfIterator(train_loader)
        device = torch.cuda.current_device()

        if args.scratch:
            print("load MAE checkpoint")
            model = ViTMAEForPreTraining.from_pretrained(args.model_name)
            state_dict = model.vit.state_dict()
        else:
            ckpt_file = os.path.join(
                args.checkpoint_dir, args.ckpt_name, args.dataset, "model.pt")
            print("load further pretrained checkpoint from", ckpt_file)
            ckpt = torch.load(ckpt_file)
            state_dict = ckpt["state_dict"]

        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        config.update(
            {
                "hidden_dropout_prob": args.dropout,
                "num_labels": num_classes
            }
        )
        model = ViTForImageClassification(config=config)
        model.vit.load_state_dict(state_dict)
        model = model.to(device)

        transform = Augmentation(224)
        transform = transform.to(device)

        no_decay = ["bias", "position_embeddings", "cls_token"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)
                           and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)
                and p.requires_grad], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_steps, args.steps)

        with tqdm(total=args.steps, leave=False, desc="Finetuning") as pbar:
            for global_step in range(1, args.steps+1):
                model.train()

                batch = next(train_iter)
                x, y = batch
                x, y = x.to(device), y.to(device)
                x = transform(x)

                outputs = model(pixel_values=x, labels=y)
                loss, logits = outputs[0], outputs[1]

                acc = accuracy(y, logits)
                model.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                pbar.set_postfix({"Steps": f"{global_step}/{args.steps}",
                                  "loss": "{0:.4f}".format(loss.item()),
                                  "acc": acc})
                pbar.update(1)

        # test loss and test accuracy
        test_acc, test_loss = test(model, test_loader, device)
        # full batch training loss and training accuracy
        print("Test loss:{:.4f}, Test accuracy: {:.4f}".format(
            test_loss, test_acc))
