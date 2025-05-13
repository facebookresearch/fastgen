# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import logging
import re
from abc import abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import torch
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open
from tqdm import tqdm

from fastgen.model import ModelArgs, RopeArgs
from fastgen.model import Transformer as Model
from fastgen.utils.tokenizer import BaseTokenizer, HfTokenizer

logger = logging.getLogger()


class BaseLoader:
    @abstractmethod
    def load(self, model: Model, show_progress: bool = True) -> None: ...

    @abstractmethod
    def load_model_args(self) -> ModelArgs: ...

    @abstractmethod
    def load_tokenizer(self) -> BaseTokenizer: ...


@dataclass
class WeightInfo:
    key: str
    tp_dim: int = -1
    shuffle: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


class HfArch:
    @staticmethod
    @abstractmethod
    def model_args(config: dict) -> ModelArgs: ...

    @staticmethod
    @abstractmethod
    def weight_info(args: ModelArgs, key: str) -> WeightInfo: ...


class LlamaForCausalLM(HfArch):
    @staticmethod
    def model_args(config: dict) -> ModelArgs:
        rope_args = RopeArgs(
            theta=config["rope_theta"],
        )
        args = ModelArgs(
            dim=config["hidden_size"],
            ffn_dim=config["intermediate_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config.get(
                "num_key_value_heads",
                config["num_attention_heads"],
            ),
            qkv_bias=False,
            norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            vocab_parallel=config["tie_word_embeddings"],
            tie_embeddings=config["tie_word_embeddings"],
            rope=rope_args,
        )
        if scaling_config := config.get("rope_scaling"):
            rope_args.use_scaled_rope = True
            rope_args.old_context_len = scaling_config[
                "original_max_position_embeddings"
            ]
            rope_args.scale_factor = scaling_config["factor"]
            rope_args.low_freq_factor = scaling_config["low_freq_factor"]
            rope_args.high_freq_factor = scaling_config["high_freq_factor"]
        # TODO: fp8 quantization
        return args

    @staticmethod
    def weight_info(args: ModelArgs, key: str) -> WeightInfo:
        match key:
            case "model.embed_tokens.weight":
                tp_dim = 0 if args.vocab_parallel else 1
                return WeightInfo("tok_embeddings.weight", tp_dim)
            case "model.norm.weight":
                return WeightInfo("norm.weight")
            case "lm_head.weight":
                return WeightInfo("output.weight", 0)

        m = re.match(r"model\.layers\.(\d+)\.(.*)\.weight", key)
        assert m is not None
        idx, module = m.groups()
        weight_map = {
            "mlp.up_proj": WeightInfo(
                f"layers.{idx}.feed_forward.w3.weight",
                tp_dim=0,
            ),
            "mlp.gate_proj": WeightInfo(
                f"layers.{idx}.feed_forward.w1.weight",
                tp_dim=0,
            ),
            "mlp.down_proj": WeightInfo(
                f"layers.{idx}.feed_forward.w2.weight",
                tp_dim=1,
            ),
            "self_attn.q_proj": WeightInfo(
                f"layers.{idx}.attention.wq.weight",
                tp_dim=0,
                shuffle=partial(
                    _rope_shuffle_linear,
                    n_heads=args.n_heads,
                ),
            ),
            "self_attn.k_proj": WeightInfo(
                f"layers.{idx}.attention.wk.weight",
                tp_dim=0,
                shuffle=partial(
                    _rope_shuffle_linear,
                    n_heads=args.n_kv_heads,
                ),
            ),
            "self_attn.v_proj": WeightInfo(
                f"layers.{idx}.attention.wv.weight",
                tp_dim=0,
            ),
            "self_attn.o_proj": WeightInfo(
                f"layers.{idx}.attention.wo.weight",
                tp_dim=1,
            ),
            "input_layernorm": WeightInfo(
                f"layers.{idx}.attention_norm.weight",
            ),
            "post_attention_layernorm": WeightInfo(
                f"layers.{idx}.ffn_norm.weight",
            ),
        }
        if module in weight_map:
            return weight_map[module]

        raise RuntimeError(f"Unexpected weight key {key}")


class Qwen2ForCausalLM(HfArch):
    @staticmethod
    def model_args(config: dict) -> ModelArgs:
        return ModelArgs(
            dim=config["hidden_size"],
            ffn_dim=config["intermediate_size"],
            n_layers=config["num_hidden_layers"],
            n_heads=config["num_attention_heads"],
            n_kv_heads=config.get(
                "num_key_value_heads",
                config["num_attention_heads"],
            ),
            qkv_bias=True,
            norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            vocab_parallel=False,
            tie_embeddings=False,
            rope=RopeArgs(theta=config["rope_theta"]),
        )

    @staticmethod
    def weight_info(args: ModelArgs, key: str) -> WeightInfo:
        m = re.match(r"model\.layers\.(\d+)\.(.*)\.bias", key)
        if m is not None:
            idx, module = m.groups()
            weight_map = {
                "self_attn.q_proj": WeightInfo(
                    f"layers.{idx}.attention.wq.bias",
                    tp_dim=0,
                    shuffle=partial(
                        _rope_shuffle_bias,
                        n_heads=args.n_heads,
                    ),
                ),
                "self_attn.k_proj": WeightInfo(
                    f"layers.{idx}.attention.wk.bias",
                    tp_dim=0,
                    shuffle=partial(
                        _rope_shuffle_bias,
                        n_heads=args.n_kv_heads,
                    ),
                ),
                "self_attn.v_proj": WeightInfo(
                    f"layers.{idx}.attention.wv.bias",
                    tp_dim=0,
                ),
            }
            if module in weight_map:
                return weight_map[module]

        return LlamaForCausalLM.weight_info(args, key)


class HfLoader(BaseLoader):
    """
    Load a Hugging Face model from a directory or
    from the Hub cache.
    """

    def __init__(self, model: str | Path) -> None:
        if isinstance(model, str):
            path = Path(model)
            if not path.is_dir():
                config_path = try_to_load_from_cache(
                    repo_id=model,
                    filename="config.json",
                )
                if not isinstance(config_path, str):
                    raise RuntimeError(f"Could not load {model}")
                path = Path(config_path).parent
            self.directory = path
        else:
            self.directory = model

        self.args: Optional[ModelArgs] = None

        config_json = self.directory / "config.json"
        if not config_json.is_file():
            raise RuntimeError(f"File {config_json} not found")

        with open(config_json) as f:
            config = json.load(f)
            config = config.get("language_config", config)
            config_arch = config["architectures"]

        self.arch: type[HfArch]
        match config_arch:
            case ["LlamaForCausalLM"] | ["MistralForCausalLM"]:
                self.arch = LlamaForCausalLM
            case ["Qwen2ForCausalLM"]:
                self.arch = Qwen2ForCausalLM
            case _:
                raise RuntimeError(
                    f"Unsupported model type {config_arch}",
                )

    def load_model_args(self) -> ModelArgs:
        with open(self.directory / "config.json") as f:
            config = json.load(f)
            config = config.get("language_config", config)
            return self.arch.model_args(config)

    def load_tokenizer(self) -> BaseTokenizer:
        return HfTokenizer(self.directory)

    def load(self, model: Model, show_progress: bool = True) -> None:
        args = self.load_model_args()
        tp_rank = model.tp_rank
        tp_size = model.tp_size
        weight_info = self.arch.weight_info

        # avoid multiple competing progress bars
        show_progress = show_progress and tp_rank == 0

        model_files = sorted(self.directory.glob("model-*-of-*.safetensors"))
        size = sum(shard.stat().st_size for shard in model_files)

        state_dict = {}
        with (
            tqdm(desc="loading", total=size, unit="B", unit_scale=True)
            if show_progress
            else nullcontext()
        ) as progress:
            for shard in model_files:
                with (
                    torch.device("cpu"),
                    safe_open(shard, framework="pt", device="cpu") as f,
                ):
                    for key in f.keys():
                        wi = weight_info(args, key)
                        w = f.get_tensor(key)
                        if progress is not None:
                            size = w.numel() * w.element_size()
                            progress.update(size)
                        if wi.shuffle is not None:
                            w = wi.shuffle(w)
                        if tp_size > 1 and wi.tp_dim >= 0:
                            chunks = torch.chunk(w, tp_size, dim=wi.tp_dim)
                            # clone to avoid leaking the full weight
                            w = chunks[tp_rank].clone()
                        state_dict[wi.key] = w

        logger.info("Transferring weights to gpu...")
        model.load_state_dict(state_dict, strict=True)


def _rope_shuffle_linear(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Permute q/k linear weights into vanilla Llama layout.
    """
    dim1, dim2 = w.shape
    return (
        w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )


def _rope_shuffle_bias(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Permute q/k bias into vanilla Llama layout.
    """
    return _rope_shuffle_linear(w.unsqueeze(1), n_heads).squeeze(1)
