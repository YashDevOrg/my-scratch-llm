import argparse
from contextlib import nullcontext
from dataclasses import asdict
import math
import os
import time
from typing import Any, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from config import ModelConfig
from model import MiniLlama
from pretrain import get_lr, load_checkpoint, save_checkpoint
from tokenizer_setup import get_or_train_tokenizer

SYNTHETIC_SFT_PAIRS = [
    ("Hello! Who are you?", "I am MiniLlama, a modern pure PyTorch language model trained from scratch."),
    ("Tell me a short story about a kitten.", "Once upon a time, a curious kitten named Bella found a secret sunny garden full of friendly butterflies."),
    ("What is 2 + 2?", "2 + 2 equals 4."),
    ("Write a poem about the night sky.", "The stars shine bright in velvet skies, watching over sleeping eyes until the golden sun arrives."),
    ("Can you help me write Python code?", "Yes! I can help you write clean, modular, and self-contained Python code using pure PyTorch."),
    ("What is machine learning?", "Machine learning is a field of artificial intelligence where algorithms learn patterns from data rather than following static rules."),
    ("Tell me a fun fact.", "Honey never spoils! Archaeologists have discovered pots of edible honey in ancient Egyptian tombs that are thousands of years old."),
    ("How do I stay healthy?", "Eat nutritious whole foods, stay physically active, drink plenty of water, and ensure you get sufficient rest every night."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Goodbye!", "Goodbye! Have a great and productive day ahead!"),
]


class SFTDataset(IterableDataset):
    """Iterable Dataset for Supervised Fine-Tuning.

    Formats each sample into:
      <|user|>\n{prompt}<|assistant|>\n{response}<|eos|>
    Masks user prompt tokens with -100 in targets so loss is only calculated on assistant responses.
    """

    def __init__(
        self,
        tokenizer: Any,
        dataset_name: str = "HuggingFaceTB/smoltalk",
        subset: Optional[str] = "everyday-conversations",
        seq_len: int = 256,
        synthetic_only: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.subset = subset
        self.seq_len = seq_len
        self.synthetic_only = synthetic_only

    def _extract_pair(self, item: dict) -> Tuple[str, str]:
        """Extract prompt and response from various dataset formats."""
        prompt = ""
        response = ""

        # 1. HuggingFaceTB/smoltalk or ChatML messages format
        if "messages" in item and isinstance(item["messages"], list):
            for m in item["messages"]:
                role = m.get("role", "")
                content = m.get("content", "").strip()
                if role == "user" and not prompt:
                    prompt = content
                elif role == "assistant" and not response:
                    response = content

        # 2. Alpaca format (instruction + input -> output)
        elif "instruction" in item and "output" in item:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            prompt = f"{inst}\n{inp}".strip() if inp else inst
            response = item.get("output", "").strip()

        # 3. Standard prompt/response format
        elif "prompt" in item and "response" in item:
            prompt = item.get("prompt", "").strip()
            response = item.get("response", "").strip()

        return prompt, response

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.synthetic_only:
            while True:
                for prompt, response in SYNTHETIC_SFT_PAIRS:
                    yield from self._process_pair(prompt, response)
        else:
            kwargs = {"split": "train", "streaming": True}
            if self.subset:
                kwargs["name"] = self.subset

            try:
                stream_ds = load_dataset(self.dataset_name, **kwargs)
                for item in stream_ds:
                    prompt, response = self._extract_pair(item)
                    if prompt and response:
                        yield from self._process_pair(prompt, response)
            except Exception as e:
                print(f"Streaming dataset error ({e}). Falling back to synthetic instructions.")
                while True:
                    for prompt, response in SYNTHETIC_SFT_PAIRS:
                        yield from self._process_pair(prompt, response)

    def _process_pair(self, prompt: str, response: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Format text with standard chat tags
        user_prefix = f"<|user|>\n{prompt}<|assistant|>\n"
        full_text = f"<|user|>\n{prompt}<|assistant|>\n{response}<|eos|>"

        prefix_ids = self.tokenizer.encode(user_prefix, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) <= 1:
            return

        # Truncate if exceeding max seq_len
        full_ids = full_ids[: self.seq_len + 1]
        chunk_tensor = torch.tensor(full_ids, dtype=torch.long)

        x = chunk_tensor[:-1]
        y = chunk_tensor[1:].clone()

        # Mask user prompt in targets so loss is only computed on assistant reply
        prefix_len = min(len(prefix_ids) - 1, len(y))
        if prefix_len > 0:
            y[:prefix_len] = -100

        # Pad to seq_len if shorter
        if len(x) < self.seq_len:
            pad_len = self.seq_len - len(x)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            x = torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])

        yield x, y


def train_sft(args: argparse.Namespace) -> None:
    """Supervised Fine-Tuning loop for MiniLlama."""
    # 1. Device configuration
    if torch.cuda.is_available():
        device = "cuda"
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler()
        print(f"Using CUDA device ({torch.cuda.get_device_name(0)}) with fp16 AMP")
    else:
        device = "cpu"
        autocast_ctx = nullcontext()
        scaler = None
        print("Using CPU device with float32")

    # 2. Tokenizer setup
    tokenizer = get_or_train_tokenizer(save_dir=args.tokenizer_dir, vocab_size=32000)
    print(f"Tokenizer loaded (vocab size: {len(tokenizer)})")

    # 3. Model setup & Pretrained checkpoint loading
    cfg = ModelConfig(vocab_size=32000, max_seq_len=max(1024, args.seq_len))
    model = MiniLlama(cfg).to(device)

    if os.path.exists(args.checkpoint_path):
        print(f"Loading pre-trained base model weights from {args.checkpoint_path}...")
        load_checkpoint(model, None, args.checkpoint_path, device=device)
    else:
        print(f"Warning: Checkpoint '{args.checkpoint_path}' not found! Initializing with base weights.")

    # 4. SFT Dataset & DataLoader
    synthetic_mode = args.smoke_test or args.synthetic
    print(
        f"Creating SFT DataLoader "
        f"(dataset='{args.dataset_name}', synthetic={synthetic_mode}, batch_size={args.batch_size}, seq_len={args.seq_len})..."
    )
    dataset = SFTDataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        subset=args.subset,
        seq_len=args.seq_len,
        synthetic_only=synthetic_mode,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader_iter = iter(dataloader)

    # 5. Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # 6. SFT Training Loop
    model.train()
    print(f"\nStarting Supervised Fine-Tuning for {args.max_steps} steps (lr={args.lr:.2e})...")
    t0 = time.time()
    losses = []

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        try:
            x, y = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            x, y = next(dataloader_iter)

        x, y = x.to(device), y.to(device)

        with autocast_ctx:
            logits, loss = model(x, targets=y)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Cosine LR schedule
        lr = get_lr(
            step=step,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            max_lr=args.lr,
            min_lr=args.lr * 0.1,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_sec = (args.batch_size * args.seq_len) / dt if dt > 0 else 0.0
        losses.append(loss.item())

        if args.smoke_test or step % 10 == 0 or step == args.max_steps:
            print(
                f"SFT Step {step:4d}/{args.max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {lr:.2e} | "
                f"{tokens_per_sec:.1f} tok/s"
            )

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(model, optimizer, cfg, step, loss.item(), args.save_path)

    print(f"\nSFT Training complete! Fine-tuned model saved to: {args.save_path}")

    # Smoke test verification
    if args.smoke_test:
        print("\n=== SFT SMOKE TEST VERIFICATION ===")
        print(f"Step 1 Loss: {losses[0]:.4f} -> Step {args.max_steps} Loss: {losses[-1]:.4f}")
        assert losses[-1] < losses[0], "SFT loss did not decrease!"
        print("SFT loss decrease verified!")
        print("Testing chat response on a sample question...")
        test_chat_single(model, tokenizer, device, "Hello! Who are you?")
        print("=== SFT SMOKE TEST PASSED ===")


def test_chat_single(
    model: MiniLlama,
    tokenizer: Any,
    device: str,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.2,
    repetition_penalty: float = 1.3,
    do_sample: bool = True,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
) -> str:
    """Run single-turn chat completion for evaluation."""
    model.eval()
    formatted = f"<|user|>\n{prompt}<|assistant|>\n"
    tokens = tokenizer.encode(formatted, add_special_tokens=False)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

    # Decode only the generated response
    gen_tokens = out[0, len(tokens) :].tolist()
    reply = tokenizer.decode(gen_tokens)

    # Cut off at stop tokens
    for stop_token in ["<|eos|>", "<|user|>", "<|assistant|>", "<|pad|>"]:
        if stop_token in reply:
            reply = reply.split(stop_token)[0]

    reply = reply.strip()
    print(f"\nUser: {prompt}")
    print(f"MiniLlama: {reply}\n")
    return reply


def chat_interactive(args: argparse.Namespace) -> None:
    """Interactive command-line chat session."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_or_train_tokenizer(save_dir=args.tokenizer_dir, vocab_size=32000)

    # Load model from SFT checkpoint if exists, else from checkpoint_path
    ckpt = args.save_path if os.path.exists(args.save_path) else args.checkpoint_path
    print(f"Loading model checkpoint from {ckpt}...")

    cfg = ModelConfig(vocab_size=32000, max_seq_len=1024)
    model = MiniLlama(cfg).to(device)

    if os.path.exists(ckpt):
        load_checkpoint(model, None, ckpt, device=device)
    else:
        print(f"Warning: Checkpoint '{ckpt}' not found! Chatting with randomly initialized weights.")

    model.eval()

    print("\n" + "=" * 60)
    print("      MiniLlama-50M Interactive Chat Mode")
    print("  Type your prompt below. Type 'exit' or 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        # Format input using chat template
        prompt_text = f"<|user|>\n{user_input}<|assistant|>\n"
        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                idx,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample,
            )

        gen_tokens = out[0, len(tokens) :].tolist()
        reply = tokenizer.decode(gen_tokens)

        # Stop at conversation boundaries
        for stop_tok in ["<|eos|>", "<|user|>", "<|assistant|>", "<|pad|>"]:
            if stop_tok in reply:
                reply = reply.split(stop_tok)[0]

        print(f"\nMiniLlama: {reply.strip()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: Supervised Fine-Tuning (SFT) & Chat for MiniLlama")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/smoltalk", help="Instruction dataset")
    parser.add_argument("--subset", type=str, default="everyday-conversations", help="Dataset subset")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/checkpoint_latest.pt", help="Base model checkpoint")
    parser.add_argument("--save_path", type=str, default="checkpoints/sft_checkpoint.pt", help="Path to save SFT checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer", help="Directory of saved tokenizer")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--max_steps", type=int, default=300, help="SFT training steps (200-500)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for SFT (1e-5 to 2e-4)")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--save_every", type=int, default=100, help="Save SFT checkpoint every N steps")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic instruction pairs")
    parser.add_argument("--smoke_test", action="store_true", help="Run 5 fast SFT steps to verify training and generation")
    parser.add_argument("--chat", action="store_true", help="Launch interactive terminal chat session")
    parser.add_argument("--temperature", type=float, default=1.2, help="Sampling temperature (default 1.2 for diversity)")
    parser.add_argument("--repetition_penalty", type=float, default=1.3, help="Repetition penalty (default 1.3)")
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sampling during generation (default True)",
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering for chat")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling (default 0.9)")
    parser.add_argument("--max_new_tokens", type=int, default=60, help="Maximum generated tokens per response")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        args.max_steps = 5
        args.batch_size = 2
        args.seq_len = 128
        args.warmup_steps = 1
        args.save_every = 5
        train_sft(args)
    elif args.chat:
        chat_interactive(args)
    else:
        train_sft(args)
