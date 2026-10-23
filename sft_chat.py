import argparse
from contextlib import nullcontext
from dataclasses import asdict
import math
import os
import random
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

# System prompt used both during SFT training and inference to anchor the model's identity
SYSTEM_PROMPT = (
    "You are MiniLlama, a helpful AI assistant that answers questions and writes Python code. "
    "Answer directly and concisely. For code requests, provide working Python code."
)

# High-priority curated conversation and Python code pairs.
# These guarantee that MiniLlama masters greetings, identity, basic questions, and fundamental Python coding.
CURATED_SFT_PAIRS = [
    # --- Greetings & Chit-Chat ---
    ("hi", "Hello! How can I help you today?"),
    ("Hi", "Hello! I am MiniLlama. How can I help you?"),
    ("hello", "Hello! I am MiniLlama. What would you like to explore today?"),
    ("Hello!", "Hello! How can I assist you today?"),
    ("hey", "Hey there! How can I help you today?"),
    ("Hey", "Hello! How can I assist you?"),
    ("good morning", "Good morning! I am ready to help. What can I do for you today?"),
    ("Good morning", "Good morning! How can I help you today?"),
    ("good evening", "Good evening! How can I assist you today?"),
    ("how are you?", "I am doing well, thank you for asking! How are you doing today?"),
    ("how are you", "I am doing great! Ready to help you with questions or Python code."),
    ("how are you doing?", "I am doing very well and ready to assist you. What are you working on?"),
    ("what are you doing?", "I am here ready to help you answer questions and write Python code."),
    ("what are you doing", "I am waiting to help you! You can ask me questions or ask for Python code."),
    ("who are you?", "I am MiniLlama, a 50M parameter language model built from scratch in pure PyTorch."),
    ("who are you", "I am MiniLlama, an AI assistant trained to answer questions and generate Python code."),
    ("what is your name?", "My name is MiniLlama."),
    ("what is your name", "My name is MiniLlama. I am an AI assistant built in pure PyTorch."),
    ("how can you help me?", "I can help by answering your questions, explaining concepts, solving math problems, and writing Python code."),
    ("how can you help me", "I can help you answer basic questions, explain computer science topics, and generate working Python code."),
    ("what can you do?", "I can converse with you, answer questions, explain concepts, and write Python programs for tasks like math, data manipulation, and algorithms."),
    ("what can you do", "I can answer questions and generate clean, working Python code."),
    ("who created you?", "I was built from scratch in pure PyTorch using modern Transformer architecture blocks."),
    ("thank you", "You're welcome! Let me know if you need anything else."),
    ("thanks", "You're welcome! Happy to help."),
    ("bye", "Goodbye! Have a great day!"),
    ("Goodbye!", "Goodbye! Reach out anytime you have questions."),

    # --- Basic Knowledge & Math ---
    ("what is 2+2 ?", "2 + 2 = 4"),
    ("what is 2 + 2?", "2 + 2 is 4."),
    ("what is 2+2", "4"),
    ("calculate 2+2", "2 + 2 = 4."),
    ("what is 5 * 5?", "5 * 5 is 25."),
    ("what is 100 / 4?", "100 / 4 equals 25."),
    ("what is machine learning?", "Machine learning is a field of artificial intelligence where algorithms learn patterns from data rather than relying on explicit hand-coded rules."),
    ("what is python?", "Python is a high-level programming language known for its simple syntax, readability, and large ecosystem for data science and AI."),
    ("what is an algorithm?", "An algorithm is a clear step-by-step sequence of instructions designed to solve a problem or perform a computation."),
    ("What is the capital of France?", "The capital of France is Paris."),

    # --- Fundamental Python Code Requests ---
    (
        "give me a python code to calculate 2+2 and print output",
        "```python\n# Calculate 2 + 2 and print output\nresult = 2 + 2\nprint(result)\n```"
    ),
    (
        "write a python code to calculate 2+2 and print output",
        "```python\nresult = 2 + 2\nprint(result)\n```"
    ),
    (
        "calculate 2+2 and print result",
        "```python\nresult = 2 + 2\nprint(\"Result:\", result)\n```"
    ),
    (
        "write code to print hello world",
        "```python\nprint(\"Hello, World!\")\n```"
    ),
    (
        "give me a python code to print hello world",
        "```python\nprint(\"Hello, World!\")\n```"
    ),
    (
        "print hello world in python",
        "```python\nprint(\"Hello, World!\")\n```"
    ),
    (
        "write a python function to add two numbers",
        "```python\ndef add(a, b):\n    return a + b\n\n# Example usage\nresult = add(5, 7)\nprint(result)  # Output: 12\n```"
    ),
    (
        "write a function to add two numbers and print result",
        "```python\ndef add(a, b):\n    return a + b\n\nprint(add(10, 20))\n```"
    ),
    (
        "write python code to check if a number is even or odd",
        "```python\ndef check_even_odd(number):\n    if number % 2 == 0:\n        return \"Even\"\n    else:\n        return \"Odd\"\n\nprint(check_even_odd(4))  # Even\nprint(check_even_odd(7))  # Odd\n```"
    ),
    (
        "write a python function to check if a number is even or odd",
        "```python\ndef is_even(n):\n    return n % 2 == 0\n\nprint(is_even(8))  # True\n```"
    ),
    (
        "write python code to reverse a string",
        "```python\ndef reverse_string(text):\n    return text[::-1]\n\nprint(reverse_string(\"hello\"))  # olleh\n```"
    ),
    (
        "write a python function to find the maximum number in a list",
        "```python\ndef find_max(numbers):\n    return max(numbers)\n\nnums = [10, 45, 2, 99, 31]\nprint(\"Maximum:\", find_max(nums))\n```"
    ),
    (
        "write a python program to calculate factorial of a number",
        "```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))  # Output: 120\n```"
    ),
    (
        "write a python program to print fibonacci numbers",
        "```python\ndef fibonacci(n):\n    sequence = [0, 1]\n    for _ in range(2, n):\n        sequence.append(sequence[-1] + sequence[-2])\n    return sequence[:n]\n\nprint(fibonacci(8))  # [0, 1, 1, 2, 3, 5, 8, 13]\n```"
    ),
    (
        "write python code to check if a number is prime",
        "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprint(is_prime(11))  # True\nprint(is_prime(4))   # False\n```"
    ),
    (
        "write python code to find the sum of all elements in a list",
        "```python\nnumbers = [1, 2, 3, 4, 5]\ntotal = sum(numbers)\nprint(\"Sum:\", total)\n```"
    ),
    (
        "write python code to sort a list of numbers",
        "```python\nnumbers = [5, 2, 9, 1, 5, 6]\nnumbers.sort()\nprint(\"Sorted:\", numbers)\n```"
    ),
    (
        "write python code to check if a string is a palindrome",
        "```python\ndef is_palindrome(s):\n    cleaned = s.lower().replace(\" \", \"\")\n    return cleaned == cleaned[::-1]\n\nprint(is_palindrome(\"racecar\"))  # True\nprint(is_palindrome(\"hello\"))    # False\n```"
    ),
    (
        "how to create a dictionary in python",
        "```python\n# Creating a dictionary and iterating over keys and values\nuser = {\"name\": \"Alice\", \"age\": 25, \"role\": \"Developer\"}\nfor key, value in user.items():\n    print(f\"{key}: {value}\")\n```"
    ),
    (
        "write python code to count vowels in a string",
        "```python\ndef count_vowels(text):\n    vowels = \"aeiouAEIOU\"\n    return sum(1 for char in text if char in vowels)\n\nprint(count_vowels(\"MiniLlama AI\"))  # Output: 6\n```"
    ),
    (
        "write python code to remove duplicates from a list",
        "```python\ndef remove_duplicates(lst):\n    return list(dict.fromkeys(lst))\n\nprint(remove_duplicates([1, 2, 2, 3, 4, 4, 5]))\n```"
    ),
    (
        "write a python program to swap two variables",
        "```python\na = 5\nb = 10\na, b = b, a\nprint(f\"a = {a}, b = {b}\")\n```"
    ),
]

# Legacy alias
SYNTHETIC_SFT_PAIRS = CURATED_SFT_PAIRS


class SFTDataset(IterableDataset):
    """Iterable Dataset for Supervised Fine-Tuning.

    Formats each sample into:
      <|user|>\n{SYSTEM_PROMPT}\n\n{prompt}<|assistant|>\n{response}<|eos|>
    Masks user prompt tokens with -100 in targets so loss is only calculated on assistant responses.
    """

    def __init__(
        self,
        tokenizer: Any,
        dataset_name: str = "tatsu-lab/alpaca",
        subset: Optional[str] = None,
        seq_len: int = 256,
        synthetic_only: bool = False,
        blend_curated: bool = True,
        blend_freq: int = 2,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        # Sanitize CLI string "None" to Python None
        if subset is not None and str(subset).strip().lower() in ("none", "null", ""):
            subset = None
        self.subset = subset
        self.seq_len = seq_len
        self.synthetic_only = synthetic_only
        self.blend_curated = blend_curated
        self.blend_freq = blend_freq

    def _extract_pair(self, item: dict) -> Tuple[str, str]:
        """Extract prompt and response from various dataset formats."""
        prompt = ""
        response = ""

        # 1. Alpaca / Code-Alpaca format (instruction + input -> output)
        if "instruction" in item and "output" in item:
            inst = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            prompt = f"{inst}\nContext: {inp}".strip() if inp else inst
            response = item.get("output", "").strip()
            # Skip empty or trivially short code responses
            if len(response) < 3:
                return prompt, response

        # 2. HuggingFaceTB/smoltalk or ChatML messages format
        elif "messages" in item and isinstance(item["messages"], list):
            for m in item["messages"]:
                role = m.get("role", "")
                content = m.get("content", "").strip()
                if role == "user" and not prompt:
                    prompt = content
                elif role == "assistant" and not response:
                    response = content

        # 3. Standard prompt/response format
        elif "prompt" in item and "response" in item:
            prompt = item.get("prompt", "").strip()
            response = item.get("response", "").strip()

        return prompt, response

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        if self.synthetic_only:
            print("Using curated instruction pairs exclusively for SFT...")
            while True:
                # Shuffle order across epochs for better generalization
                shuffled_pairs = list(CURATED_SFT_PAIRS)
                random.shuffle(shuffled_pairs)
                for prompt, response in shuffled_pairs:
                    yield from self._process_pair(prompt, response)
        else:
            kwargs = {"split": "train", "streaming": True}
            if self.subset:
                kwargs["name"] = self.subset

            print(f"Streaming live dataset '{self.dataset_name}' (subset={self.subset}, blend_curated={self.blend_curated})...")
            stream_ds = load_dataset(self.dataset_name, **kwargs)
            curated_cycle = list(CURATED_SFT_PAIRS)
            curated_idx = 0
            count = 0

            for item in stream_ds:
                prompt, response = self._extract_pair(item)
                if prompt and response:
                    yield from self._process_pair(prompt, response)
                    count += 1

                    # Interleave curated core greetings & basic Python tasks regularly
                    if self.blend_curated and count % self.blend_freq == 0:
                        cur_p, cur_r = curated_cycle[curated_idx % len(curated_cycle)]
                        curated_idx += 1
                        yield from self._process_pair(cur_p, cur_r)

    def _process_pair(self, prompt: str, response: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Format with system prompt + chat tags.
        # The system prompt anchors identity during both training and inference.
        sys_prefix = f"<|user|>\n{SYSTEM_PROMPT}\n\n{prompt}<|assistant|>\n"
        full_text  = f"<|user|>\n{SYSTEM_PROMPT}\n\n{prompt}<|assistant|>\n{response}<|eos|>"

        prefix_ids = self.tokenizer.encode(sys_prefix, add_special_tokens=False)
        full_ids   = self.tokenizer.encode(full_text,  add_special_tokens=False)

        if len(full_ids) <= 1:
            return

        # Truncate if exceeding max seq_len
        full_ids = full_ids[: self.seq_len + 1]
        chunk_tensor = torch.tensor(full_ids, dtype=torch.long)

        x = chunk_tensor[:-1]
        y = chunk_tensor[1:].clone()

        # Mask user prompt + system prompt tokens; loss only on assistant reply
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
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()
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
    synthetic_mode = args.smoke_test or args.synthetic or args.curated_only
    print(
        f"Creating SFT DataLoader "
        f"(dataset='{args.dataset_name}', curated_only={synthetic_mode}, "
        f"blend_curated={args.blend_curated}, batch_size={args.batch_size}, seq_len={args.seq_len})..."
    )
    dataset = SFTDataset(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        subset=args.subset,
        seq_len=args.seq_len,
        synthetic_only=synthetic_mode,
        blend_curated=args.blend_curated,
        blend_freq=args.blend_freq,
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
    print(f"\nStarting Supervised Fine-Tuning for {args.max_steps} steps (lr={args.lr:.2e}, grad_accum={args.grad_accum_steps})...")
    t0 = time.time()
    losses = []

    for step in range(1, args.max_steps + 1):
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
                loss_scaled = loss / args.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
        effective_tokens = args.batch_size * args.seq_len * args.grad_accum_steps
        tokens_per_sec = effective_tokens / dt if dt > 0 else 0.0
        avg_step_loss = accum_loss / args.grad_accum_steps
        losses.append(avg_step_loss)

        if args.smoke_test or step % 10 == 0 or step == args.max_steps:
            print(
                f"SFT Step {step:4d}/{args.max_steps} | "
                f"Loss: {avg_step_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"{tokens_per_sec:.1f} tok/s"
            )

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(model, optimizer, cfg, step, avg_step_loss, args.save_path)

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
    max_new_tokens: int = 64,
    temperature: float = 0.3,
    repetition_penalty: float = 1.15,
    do_sample: bool = True,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
) -> str:
    """Run single-turn chat completion for evaluation."""
    model.eval()
    formatted = f"<|user|>\n{SYSTEM_PROMPT}\n\n{prompt}<|assistant|>\n"
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

    ckpt = args.checkpoint_path
    print(f"Loading model checkpoint from {ckpt}...")

    cfg = ModelConfig(vocab_size=32000, max_seq_len=1024)
    model = MiniLlama(cfg).to(device)

    if os.path.exists(ckpt):
        checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if hasattr(model, "lm_head") and hasattr(model, "tok_embeddings"):
            model.lm_head.weight = model.tok_embeddings.weight
        step = checkpoint.get("step", 0)
        loss = checkpoint.get("loss", 0.0)
        print(f"Loaded checkpoint from: {ckpt} (step {step}, loss: {loss:.4f})")
    else:
        print(f"Warning: Checkpoint '{ckpt}' not found! Chatting with randomly initialized weights.")

    model.eval()

    print("\n" + "=" * 60)
    print("      MiniLlama-50M Interactive Chat Mode")
    print("  Type your prompt below. Type 'exit' or 'quit' to exit.")
    print(f"  (temperature={args.temperature}, top_p={args.top_p}, rep_penalty={args.repetition_penalty})")
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

        # Inject system prompt so the model knows its role
        prompt_text = f"<|user|>\n{SYSTEM_PROMPT}\n\n{user_input}<|assistant|>\n"
        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(
                prompt_tensor,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                do_sample=args.do_sample if args.temperature > 0.0 else False,
            )

        gen_tokens = out[0, len(tokens):].tolist()
        reply = tokenizer.decode(gen_tokens)

        # Stop at conversation boundaries
        for stop_tok in ["<|eos|>", "<|user|>", "<|assistant|>", "<|pad|>"]:
            if stop_tok in reply:
                reply = reply.split(stop_tok)[0]

        reply = reply.strip()
        if not reply:
            reply = "[No response generated. Please try another prompt.]"

        print(f"\nMiniLlama: {reply}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: Supervised Fine-Tuning (SFT) & Chat for MiniLlama")
    parser.add_argument("--dataset_name", type=str, default="iamtarun/python_code_instructions_18k_alpaca", help="Instruction dataset")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset (leave None for Alpaca)")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/checkpoint_latest.pt",
        help="Base pretrained model checkpoint to load",
    )
    parser.add_argument("--save_path", type=str, default="checkpoints/sft_final.pt", help="Path to save SFT checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer", help="Directory of saved tokenizer")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for SFT (16 fits well on T4)")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--max_steps", type=int, default=500, help="SFT training steps (300-800)")
    parser.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate for SFT")
    parser.add_argument("--warmup_steps", type=int, default=30, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--save_every", type=int, default=100, help="Save SFT checkpoint every N steps")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic instruction pairs")
    parser.add_argument("--curated_only", action="store_true", help="Train exclusively on curated greetings and Python code pairs")
    parser.add_argument("--blend_curated", action=argparse.BooleanOptionalAction, default=True, help="Interleave curated pairs in streaming dataset")
    parser.add_argument("--blend_freq", type=int, default=2, help="Frequency to inject curated pairs into live stream")
    parser.add_argument("--smoke_test", action="store_true", help="Run 5 fast SFT steps to verify training and generation")
    parser.add_argument("--chat", action="store_true", help="Launch interactive terminal chat session")
    parser.add_argument("--redo_sft", action="store_true", help="Re-run SFT from --checkpoint_path")
    # Low temperature (0.3) produces much crisper code and stops hallucinations
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (0.2 to 0.4 best for code and chat)")
    parser.add_argument("--repetition_penalty", type=float, default=1.15, help="Repetition penalty (1.1 to 1.2)")
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable sampling during generation (default True)",
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering for chat")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling (default 0.9)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum generated tokens per response")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        args.max_steps = 5
        args.batch_size = 2
        args.grad_accum_steps = 1
        args.seq_len = 128
        args.warmup_steps = 1
        args.save_every = 5
        train_sft(args)
    elif args.chat:
        chat_interactive(args)
    elif args.redo_sft:
        print("Re-running SFT from base checkpoint (ignoring existing sft_checkpoint.pt)...")
        train_sft(args)
    else:
        train_sft(args)
