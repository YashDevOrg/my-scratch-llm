from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2
    max_seq_len: int = 1024
    multiple_of: int = 256
    norm_eps: float = 1e-5
    dropout: float = 0.0
