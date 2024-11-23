from cycling_utils import TimestampedTimer

timer = TimestampedTimer("Imported TimestampedTimer")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse
import os
import socket
import yaml
import time
import math

from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    AtomicDirectory,
    atomic_torch_save,
)

from utils.optimizers import Lamb
from utils.datasets import EVAL_HDF_Dataset
from model import Model

timer.report("Completed imports")

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default="/root/chess-hackathon-4/model_config.yaml")
    parser.add_argument("--save-dir", help="save checkpoint path", type=Path, default=os.environ.get("OUTPUT_PATH"))
    parser.add_argument("--load-path", type=Path, default="/root/chess-hackathon-4/checkpoint.pt")
    parser.add_argument("--bs", type=int, default=128)  # Increased batch size
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--wd", type=float, default=0.005)
    parser.add_argument("--ws", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)  # For gradient clipping
    return parser

def logish_transform(data):
    '''Zero-symmetric log-transformation.'''
    reflector = -1 * (data < 0).to(torch.int8)
    return reflector * torch.log(torch.abs(data) + 1)

def spearmans_rho(seq_a, seq_b):
    '''Spearman's rank correlation coefficient'''
    assert len(seq_a) == len(seq_b), "ERROR: Sortables must be equal length."
    index = range(len(seq_a))
    sorted_by_a = [t[0] for t in sorted(zip(index, seq_a, seq_b), key=lambda t: t[1])]
    sorted_by_b = [t[0] for t in sorted(zip(index, seq_a, seq_b), key=lambda t: t[2])]
    return 1 - 6 * sum([(sorted_by_a.index(i) - sorted_by_b.index(i))**2 for i in index]) / (len(seq_a) * (len(seq_a)**2 - 1))

def main(args, timer):
    # Initialize distributed training
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    args.device_id = int(os.environ["LOCAL_RANK"])
    args.is_master = rank == 0
    torch.cuda.set_device(args.device_id)
    torch.autograd.set_detect_anomaly(True)

    if args.device_id == 0:
        hostname = socket.gethostname()
        print("Hostname:", hostname)
        print(f"TrainConfig: {args}")
    timer.report("Setup for distributed training")

    # Initialize saver
    saver = AtomicDirectory(args.save_dir)
    timer.report("Validated checkpoint path")

    # Load dataset
    data_path = "/data"
    dataset = EVAL_HDF_Dataset(data_path)
    timer.report(f"Initialized dataset with {len(dataset):,} Board Evaluations.")

    # Split dataset
    random_generator = torch.Generator().manual_seed(42)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=random_generator)

    # Create dataloaders
    train_sampler = InterruptableDistributedSampler(train_dataset)
    test_sampler = InterruptableDistributedSampler(test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, sampler=train_sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=args.bs, sampler=test_sampler)
    timer.report("Prepared dataloaders")

    # Initialize model
    model_config = yaml.safe_load(open(args.model_config))
    if args.device_id == 0:
        print(f"ModelConfig: {model_config}")
    model_config["device"] = 'cuda'
    model = Model(**model_config)
    model = model.to(args.device_id)

    # Count parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    timer.report(f"Initialized model with {params:,} params, moved to device")

    # Wrap model for distributed training
    model = DDP(model, device_ids=[args.device_id])
    timer.report("Prepared model for distributed training")

    # Initialize training components
    loss_fn = nn.MSELoss()
    optimizer = Lamb(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()
    metrics = {"train": MetricsTracker(), "test": MetricsTracker()}

    # Load checkpoint if exists
    checkpoint_path = None
    local_resume_path = os.path.join(args.save_dir, saver.symlink_name)
    if os.path.islink(local_resume_path):
        checkpoint = os.path.join(os.readlink(local_resume_path), "checkpoint.pt")
        if os.path.isfile(checkpoint):
            checkpoint_path = checkpoint
    elif args.load_path:
        if os.path.isfile(args.load_path):
            checkpoint_path = args.load_path

    if checkpoint_path:
        if args.is_master:
            timer.report(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.device_id}")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_dataloader.sampler.load_state_dict(checkpoint["train_sampler"])
        test_dataloader.sampler.load_state_dict(checkpoint["test_sampler"])
        metrics = checkpoint["metrics"]
        timer = checkpoint["timer"]
        timer.start_time = time.time()
        timer.report("Retrieved saved checkpoint")

    # Training loop
    for epoch in range(train_dataloader.sampler.epoch, 10_000):
        with train_dataloader.sampler.in_epoch(epoch):
            timer.report(f"Training epoch {epoch}")
            train_batches_per_epoch = len(train_dataloader)
            train_steps_per_epoch = math.ceil(train_batches_per_epoch / args.grad_accum)
            optimizer.zero_grad()
            model.train()

            for boards, scores in train_dataloader:
                # Batch tracking
                batch = train_dataloader.sampler.progress // train_dataloader.batch_size
                is_save_batch = (batch + 1) % args.save_steps == 0
                is_accum_batch = (batch + 1) % args.grad_accum == 0
                is_last_batch = (batch + 1) == train_batches_per_epoch

                # if args.is_master and batch % 500 == 0:
                #     print("\nData Sample:")
                #     print(f"Board shape: {boards.shape}")
                #     print(f"First board in batch:\n{boards[0]}")  # Shows piece positions
                #     print(f"First 5 scores: {scores[:5].tolist()}")  # Shows evaluation scores
                #     print("-" * 50)

                # Prepare checkpoint directory
                if (is_save_batch or is_last_batch) and args.is_master:
                    checkpoint_directory = saver.prepare_checkpoint_directory()

                # Process batch
                scores = logish_transform(scores)
                boards, scores = boards.to(args.device_id), scores.to(args.device_id)

                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    logits = model(boards)
                    loss = loss_fn(logits, scores) / args.grad_accum

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                train_dataloader.sampler.advance(len(scores))

                # Calculate metrics
                rank_corr = spearmans_rho(logits.detach(), scores)
                metrics["train"].update({
                    "examples_seen": len(scores),
                    "accum_loss": loss.item() * args.grad_accum,
                    "rank_corr": rank_corr
                })

                if is_accum_batch or is_last_batch:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Update learning rate
                    scheduler.step()
                    
                    # Report metrics
                    step = batch // args.grad_accum
                    metrics["train"].reduce()
                    rpt = metrics["train"].local
                    avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                    rpt_rank_corr = 100 * rpt["rank_corr"] / rpt["examples_seen"]
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    report = f"""\
Epoch [{epoch:,}] Step [{step:,}/{train_steps_per_epoch:,}] Batch [{batch:,}/{train_batches_per_epoch:,}] \
Lr: [{current_lr:,.6f}], Avg Loss [{avg_loss:,.3f}], Rank Corr.: [{rpt_rank_corr:,.3f}%], \
Examples: {rpt['examples_seen']:,.0f}"""
                    timer.report(report)
                    metrics["train"].reset_local()

                # Save checkpoint
                if (is_save_batch or is_last_batch) and args.is_master:
                    atomic_torch_save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "train_sampler": train_dataloader.sampler.state_dict(),
                            "test_sampler": test_dataloader.sampler.state_dict(),
                            "metrics": metrics,
                            "timer": timer
                        },
                        os.path.join(checkpoint_directory, "checkpoint.pt"),
                    )
                    saver.atomic_symlink(checkpoint_directory)

        # Evaluation loop
        with test_dataloader.sampler.in_epoch(epoch):
            timer.report(f"Testing epoch {epoch}")
            test_batches_per_epoch = len(test_dataloader)
            model.eval()

            with torch.no_grad():
                for boards, scores in test_dataloader:
                    batch = test_dataloader.sampler.progress // test_dataloader.batch_size
                    is_save_batch = (batch + 1) % args.save_steps == 0
                    is_last_batch = (batch + 1) == test_batches_per_epoch

                    if (is_save_batch or is_last_batch) and args.is_master:
                        checkpoint_directory = saver.prepare_checkpoint_directory()

                    scores = logish_transform(scores)
                    boards, scores = boards.to(args.device_id), scores.to(args.device_id)

                    with torch.cuda.amp.autocast():
                        logits = model(boards)
                        loss = loss_fn(logits, scores)
                    
                    test_dataloader.sampler.advance(len(scores))
                    rank_corr = spearmans_rho(logits, scores)

                    metrics["test"].update({
                        "examples_seen": len(scores),
                        "accum_loss": loss.item() * args.grad_accum,
                        "rank_corr": rank_corr
                    })

                    if is_last_batch:
                        metrics["test"].reduce()
                        rpt = metrics["test"].local
                        avg_loss = rpt["accum_loss"] / rpt["examples_seen"]
                        rpt_rank_corr = 100 * rpt["rank_corr"] / rpt["examples_seen"]
                        report = f"Epoch [{epoch}] Evaluation, Avg Loss [{avg_loss:,.3f}], Rank Corr. [{rpt_rank_corr:,.3f}%]"
                        timer.report(report)

                    if (is_save_batch or is_last_batch) and args.is_master:
                        atomic_torch_save(
                            {
                                "model": model.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "train_sampler": train_dataloader.sampler.state_dict(),
                                "test_sampler": test_dataloader.sampler.state_dict(),
                                "metrics": metrics,
                                "timer": timer
                            },
                            os.path.join(checkpoint_directory, "checkpoint.pt"),
                        )
                        saver.atomic_symlink(checkpoint_directory)

timer.report("Defined functions")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args, timer)