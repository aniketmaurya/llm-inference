import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
from dotenv import load_dotenv
from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import EmptyInitOnDevice

load_dotenv()

WEIGHTS_PATH = os.environ.get("WEIGHTS")


@torch.no_grad()
def _generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    max_seq_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (B, T) with indices of the prompt sequence.
        max_new_tokens: The number of new tokens to generate.
        max_seq_length: The maximum sequence length allowed.
        temperature: Scales the predicted logits by 1 / temperature
        top_k: If specified, only sample among the tokens with the k highest probabilities
    """
    # create an empty tensor of the expected final shape and fill in the current tokens
    B, T = idx.shape
    T_new = T + max_new_tokens
    empty = torch.empty(B, T_new, dtype=idx.dtype, device=idx.device)
    empty[:, :T] = idx
    idx = empty

    # generate max_new_tokens tokens
    for t in range(T, T_new):
        # ignore the not-filled-yet tokens
        idx_cond = idx[:, :t]
        # if the sequence context is growing too long we must crop it at max_seq_length
        idx_cond = idx_cond if T <= max_seq_length else idx_cond[:, -max_seq_length:]

        # forward
        logits = model(idx_cond)
        logits = logits[:, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # concatenate the new column
        idx[:, t:] = idx_next

    return idx


class LLaMAInference:
    def __init__(
        self,
        model_size: str = "7B",
        dtype: Optional[str] = None,
        quantize: Optional[str] = None,
        checkpoint_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
        accelerator: str = "auto",
        devices: int = 1,
    ) -> None:
        self.fabric = fabric = L.Fabric(accelerator=accelerator, devices=devices)

        if not checkpoint_path and WEIGHTS_PATH:
            checkpoint_path = f"{WEIGHTS_PATH}/{model_size}/state_dict.pth"
            tokenizer_path = f"{WEIGHTS_PATH}/tokenizer.model"

        if dtype is not None:
            dt = getattr(torch, dtype, None)
            if not isinstance(dt, torch.dtype):
                raise ValueError(f"{dtype} is not a valid dtype.")
            dtype = dt

        with EmptyInitOnDevice(
            device=fabric.device, dtype=dtype, quantization_mode=quantize
        ):
            print("Loading model ...", file=sys.stderr)
            t0 = time.time()
            model = LLaMA.from_name(model_size)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
            print(
                f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr
            )

        self.eval()
        self.model = fabric.setup_module(model)
        self.tokenizer = Tokenizer(tokenizer_path)

    def eval(self):
        self.model.eval()

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        top_k: int = 200,
        temperature: float = 0.8,
    ) -> str:
        return self.generate(prompt, max_new_tokens, top_k, temperature)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        top_k: int = 200,
        temperature: float = 0.8,
    ) -> str:
        encoded_prompt = self.tokenizer.encode(
            prompt, bos=True, eos=False, device=self.fabric.device
        )
        encoded_prompt = encoded_prompt[None, :]  # add batch dimension
        y = _generate(
            self.model,
            encoded_prompt,
            max_new_tokens,
            self.model.config.block_size,  # type: ignore[union-attr,arg-type]
            temperature=temperature,
            top_k=top_k,
        )[0]
        return self.tokenizer.decode(y)

    def load_state_dict_from_path(self, checkpoint_path: Union[Path, str]):
        checkpoints = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoints, strict=False)

    def load_lora_weights(self, checkpoint_path: str):
        self.load_state_dict_from_path(checkpoint_path)

    def load_adapters_weights(self, checkpoint_path: str):
        self.load_state_dict_from_path(checkpoint_path)
