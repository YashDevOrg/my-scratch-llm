import os
from typing import Optional

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]


def get_or_train_tokenizer(
    save_dir: str = "tokenizer",
    vocab_size: int = 32000,
    dataset_name: str = "roneneldan/TinyStories",
    num_train_docs: int = 10000,
) -> PreTrainedTokenizerFast:
    """Load a trained BPE tokenizer from disk if it exists, or train a new one from scratch.

    Args:
        save_dir: Directory to save/load the tokenizer files.
        vocab_size: Target vocabulary size (default 32000).
        dataset_name: Hugging Face dataset to stream for training (default roneneldan/TinyStories).
        num_train_docs: Number of documents to stream for training (default 10000).

    Returns:
        PreTrainedTokenizerFast: Hugging Face compatible fast tokenizer instance.
    """
    tokenizer_json = os.path.join(save_dir, "tokenizer.json")
    if os.path.isdir(save_dir) and os.path.exists(tokenizer_json):
        print(f"Loading tokenizer from existing directory: {save_dir}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(save_dir)
        tokenizer.pad_token = "<|pad|>"
        tokenizer.bos_token = "<|bos|>"
        tokenizer.eos_token = "<|eos|>"
        return tokenizer

    print(f"Training BPE tokenizer from scratch (vocab_size={vocab_size})...")
    os.makedirs(save_dir, exist_ok=True)

    # Initialize raw ByteLevel BPE tokenizer
    raw_tokenizer = Tokenizer(BPE(unk_token=None))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Streaming {num_train_docs} documents from '{dataset_name}'...")
    stream_ds = load_dataset(dataset_name, split="train", streaming=True)

    def batch_iterator(batch_size: int = 1000):
        batch = []
        for i, item in enumerate(stream_ds):
            if i >= num_train_docs:
                break
            text = item.get("text", "")
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    raw_tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=num_train_docs)

    # Wrap into PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        clean_up_tokenization_spaces=False,
    )
    fast_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|user|>", "<|assistant|>"]}
    )

    print(f"Saving tokenizer to {save_dir}...")
    fast_tokenizer.save_pretrained(save_dir)
    print("Tokenizer successfully trained and saved!")

    return fast_tokenizer


if __name__ == "__main__":
    tokenizer = get_or_train_tokenizer(save_dir="tokenizer", vocab_size=32000)
    print(f"Vocab size: {len(tokenizer)}")
    for token in SPECIAL_TOKENS:
        print(f"Token '{token}' -> ID {tokenizer.convert_tokens_to_ids(token)}")
