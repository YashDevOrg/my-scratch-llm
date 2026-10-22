import math
from typing import Any, Iterator, Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from tokenizer_setup import get_or_train_tokenizer


class StreamingTokenDataset(IterableDataset):
    """PyTorch IterableDataset for autoregressive pre-training.

    Streams documents, tokenizes text, and continuously packs token chunks into
    (x, y) tensor pairs of length seq_len.
    """

    def __init__(
        self,
        tokenizer: Any,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        subset: Optional[str] = "sample-10BT",
        seq_len: int = 1024,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.subset = subset
        self.seq_len = seq_len
        self.eos_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        kwargs = {"split": "train", "streaming": True}
        if self.subset:
            kwargs["name"] = self.subset

        stream_ds = load_dataset(self.dataset_name, **kwargs)

        # Support multi-worker DataLoader if num_workers > 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            stream_ds = stream_ds.shard(
                num_shards=worker_info.num_workers, index=worker_info.id
            )

        buffer = []
        target_chunk_len = self.seq_len + 1

        for sample in stream_ds:
            text = sample.get("text", "")
            if not text:
                continue

            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids.append(self.eos_token_id)
            buffer.extend(token_ids)

            while len(buffer) >= target_chunk_len:
                chunk = buffer[:target_chunk_len]
                buffer = buffer[target_chunk_len:]

                chunk_tensor = torch.tensor(chunk, dtype=torch.long)
                x = chunk_tensor[:-1]
                y = chunk_tensor[1:]
                yield x, y


def create_pretrain_dataloader(
    tokenizer: Any,
    batch_size: int = 4,
    seq_len: int = 1024,
    dataset_name: str = "roneneldan/TinyStories",
    subset: Optional[str] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Wrap StreamingTokenDataset inside a PyTorch DataLoader."""
    dataset = StreamingTokenDataset(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        subset=subset,
        seq_len=seq_len,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


if __name__ == "__main__":
    print("--- Stage 3 Smoke Test: Tokenizer and Streaming Data Pipeline ---")

    # 1. Train or load tokenizer
    tokenizer = get_or_train_tokenizer(save_dir="tokenizer", vocab_size=32000)
    print(f"\nTokenizer vocab size: {len(tokenizer)}")
    print("Special token mapping:")
    for tok in ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]:
        print(f"  {tok} -> ID {tokenizer.convert_tokens_to_ids(tok)}")

    # 2. Test encoding / decoding sample with conversation tags
    sample_text = (
        "<|user|>\nTell me a short story about a brave puppy.<|assistant|>\n"
        "Once upon a time, a little golden puppy named Barnaby explored the enchanted woods.<|eos|>"
    )
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nSample Input Text:\n{sample_text}")
    print(f"Encoded token IDs ({len(encoded)} tokens): {encoded[:15]}...")
    print(f"Decoded Text:\n{decoded}")
    assert "<|user|>" in decoded and "<|assistant|>" in decoded and "<|eos|>" in decoded, (
        "Special tokens were not preserved properly during encode/decode!"
    )
    print("Special token encoding/decoding verified!")

    # 3. Test StreamingTokenDataset and MiniLlama forward pass
    from config import ModelConfig
    from model import MiniLlama

    test_batch_size = 2
    test_seq_len = 128
    print(
        f"\nCreating DataLoader for StreamingTokenDataset "
        f"(dataset='roneneldan/TinyStories', batch_size={test_batch_size}, seq_len={test_seq_len})..."
    )
    dataloader = create_pretrain_dataloader(
        tokenizer=tokenizer,
        batch_size=test_batch_size,
        seq_len=test_seq_len,
        dataset_name="roneneldan/TinyStories",
        subset=None,
    )

    print("Fetching 1 batch (x, y) from stream...")
    batch_iter = iter(dataloader)
    x, y = next(batch_iter)
    print(f"Batch x shape: {x.shape}, dtype: {x.dtype}")
    print(f"Batch y shape: {y.shape}, dtype: {y.dtype}")
    assert x.shape == (test_batch_size, test_seq_len), f"Unexpected x shape: {x.shape}"
    assert y.shape == (test_batch_size, test_seq_len), f"Unexpected y shape: {y.shape}"

    # Sample preview
    preview = tokenizer.decode(x[0, :35].tolist())
    print(f"First sequence preview (first 35 tokens): '{preview}'")

    print("\nInitializing MiniLlama (~50M) and running forward pass with streaming batch...")
    cfg = ModelConfig(vocab_size=32000, max_seq_len=1024)
    model = MiniLlama(cfg)
    model.eval()

    with torch.no_grad():
        logits, loss = model(x, targets=y)

    expected_loss = math.log(cfg.vocab_size)
    print(f"Logits tensor shape: {logits.shape}")
    print(f"Forward pass loss: {loss.item():.4f} (theoretical random loss ~= {expected_loss:.4f})")
    assert logits.shape == (test_batch_size, test_seq_len, cfg.vocab_size)
    assert loss is not None and not torch.isnan(loss)

    print("\n--- Pipeline Smoke Test Passed Successfully! ---")
