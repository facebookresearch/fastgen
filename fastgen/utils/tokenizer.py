# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
from abc import abstractmethod
from pathlib import Path
from typing import Literal, Optional, TypeAlias, TypedDict

from jinja2 import Template
from tokenizers import Tokenizer


class Message(TypedDict):
    # ipython is used by llama models
    role: Literal["system", "user", "assistant", "ipython"]
    content: str


Dialog: TypeAlias = list[Message]


class BaseTokenizer:
    @abstractmethod
    def encode(self, content: str) -> list[int]: ...
    @abstractmethod
    def decode(self, tokens: list[int]) -> str: ...

    @abstractmethod
    def str_to_id(self, s: str) -> int: ...
    @abstractmethod
    def id_to_str(self, id: int) -> Optional[str]: ...

    @abstractmethod
    def encode_dialog(
        self,
        dialog: Dialog,
        for_completion: bool = True,
    ) -> list[int]: ...

    @property
    @abstractmethod
    def stop_tokens(self) -> list[int]: ...

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...


class HfTokenizer(BaseTokenizer):
    """
    Use the Hugging Face tokenizer library to build
    a tokenizer from a model path (e.g., cloned with
    git lfs).
    """

    def __init__(self, directory: str | Path) -> None:
        if isinstance(directory, str):
            directory = Path(directory)

        tokenizer = directory / "tokenizer.json"
        if not tokenizer.is_file():
            raise RuntimeError(f"Could not find {tokenizer}")

        tokenizer_config = directory / "tokenizer_config.json"
        if not tokenizer_config.is_file():
            raise RuntimeError(f"Could not find {tokenizer_config}")

        self._tk = Tokenizer.from_file(str(tokenizer))
        with open(tokenizer_config) as f:
            self._config = json.load(f)

    def encode(self, content: str) -> list[int]:
        return self._tk.encode(content, add_special_tokens=False).ids

    def decode(self, tokens: list[int]) -> str:
        return self._tk.decode(tokens, skip_special_tokens=False)

    def id_to_str(self, id: int) -> Optional[str]:
        return self._tk.id_to_token(id)

    def str_to_id(self, s: str) -> int:
        id = self._tk.token_to_id(s)
        assert id is not None
        return id

    def encode_dialog(
        self,
        dialog: Dialog,
        for_completion: bool = True,
    ) -> list[int]:
        """
        Note: Will raise if the tokenizer does not support
        chat format.
        """
        tmpl = Template(self._config["chat_template"])
        bos_token = self._config["bos_token"]
        if isinstance(bos_token, dict):
            bos_token = bos_token["content"]
        eos_token = self._config.get("eos_token")
        chat = tmpl.render(
            messages=dialog,
            add_generation_prompt=for_completion,
            bos_token=bos_token,
            eos_token=eos_token,
        )
        return self.encode(chat)

    @property
    def stop_tokens(self) -> list[int]:
        eos_token = self._config.get("eos_token")
        if not eos_token:
            return []
        if isinstance(eos_token, dict):
            eos_token = eos_token["content"]
        return [self.str_to_id(eos_token)]

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()
