import argparse
from contextlib import nullcontext
from dataclasses import asdict
import math
import os
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn

from config import ModelConfig
from dataset import create_pretrain_dataloader
from model import MiniLlama
from tokenizer_setup import get_or_train_tokenizer


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate scheduler with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: ModelConfig,
    step: int,
    loss: float,
    checkpoint_path: str,
) -> None:
    """Save model weights, optimizer state, config, step counter, and loss."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": asdict(config),
        "step": step,
        "loss": loss,
    }
    torch.save(state, checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path} (step {step}, loss: {loss:.4f})")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: str,
) -> Tuple[int, float]:
    """Load model state and optional optimizer state from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    # Ensure weight tying is preserved
    if hasattr(model, "lm_head") and hasattr(model, "tok_embeddings"):
        model.lm_head.weight = model.tok_embeddings.weight
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", 0.0)
    print(f"Loaded checkpoint from: {checkpoint_path} (resuming at step {step}, loss: {loss:.4f})")
    return step, loss


def generate_sample(
    model: MiniLlama,
    tokenizer,
    device: str,
    prompt: str = "Once upon a time, there was a little",
    max_new_tokens: int = 30,
) -> str:
    """Generate sample continuation from prompt."""
    model.eval()
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    text = tokenizer.decode(out[0].tolist())
    model.train()
    return text


def train(args: argparse.Namespace) -> None:
    # 1. Device and precision configuration
    if torch.cuda.is_available():
        device = "cuda"
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)}) with fp16 AMP")
    else:
        device = "cpu"
        autocast_ctx = nullcontext()
        scaler = None
        print("Device: CPU with float32")

    # 2. Tokenizer and DataLoader
    tokenizer = get_or_train_tokenizer(save_dir="tokenizer", vocab_size=32000)
    print(f"Tokenizer loaded with vocab size: {len(tokenizer)}")

    print(
        f"Initializing streaming dataset '{args.dataset_name}' "
        f"(batch_size={args.batch_size}, seq_len={args.seq_len}, grad_accum_steps={args.grad_accum_steps})..."
    )
    dataloader = create_pretrain_dataloader(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dataset_name=args.dataset_name,
        subset=args.subset,
    )
    dataloader_iter = iter(dataloader)

    # 3. Model setup
    cfg = ModelConfig(
        vocab_size=32000,
        max_seq_len=max(1024, args.seq_len),
    )
    model = MiniLlama(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MiniLlama model initialized: {total_params / 1e6:.2f}M trainable parameters")

    # 4. AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # 5. Checkpoint Resume
    start_step = 0
    ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    if args.resume and os.path.exists(ckpt_path):
        start_step, _ = load_checkpoint(model, optimizer, ckpt_path, device)
        if start_step >= args.max_steps:
            print(f"Checkpoint step {start_step} is already >= max_steps {args.max_steps}. Nothing to train.")
            return

    # 6. Training loop
    model.train()
    print(f"\nStarting pre-training from step {start_step + 1} to {args.max_steps}...")
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    t0 = time.time()
    losses = []

    for step in range(start_step + 1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            try:
                x, y = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                x, y = next(dataloader_iter)

            x, y = x.to(device), y.to(device)

            with autocast_ctx:
                logits, loss = model(x, targets=y)
                loss = loss / args.grad_accum_steps

            accum_loss += loss.item()

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Gradient clipping and optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update learning rate
        lr = get_lr(
            step=step,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_sec = tokens_per_step / dt if dt > 0 else 0.0
        losses.append(accum_loss)

        # Log every step during small runs/tests, or every 10 steps in full runs
        if args.max_steps <= 20 or step % 10 == 0 or step == args.max_steps:
            print(
                f"Step {step:4d}/{args.max_steps} | "
                f"Loss: {accum_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"{tokens_per_sec:.1f} tok/s"
            )

        # Periodic checkpoint saving and sample generation
        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(model, optimizer, cfg, step, accum_loss, ckpt_path)
            print(f"\n--- Sample Generation at Step {step} ---")
            sample_text = generate_sample(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=args.eval_prompt,
                max_new_tokens=30,
            )
            print(f"Generated text:\n\"{sample_text}\"\n")

    print(f"Training completed through step {args.max_steps}!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: The Pre-Training Engine for MiniLlama")
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories", help="Dataset name to stream")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset/config name")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per forward pass")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=2000, help="Total training steps")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--max_lr", type=float, default=6e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW weight decay")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument(
        "--eval_prompt",
        type=str,
        default="Once upon a time, there was a little",
        help="Prompt to evaluate text generation during checkpointing",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from checkpoint_latest.pt if it exists in checkpoint_dir",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
