import json
import os
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Union

import lightning as L
import torch
from dotenv import load_dotenv
from lightning.fabric.strategies import FSDPStrategy
from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.adapter_v2 import add_adapter_v2_parameters_to_linear_layers
from lit_gpt.model import Block
from lit_gpt.utils import check_valid_checkpoint_dir, lazy_load, quantization
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

load_dotenv()

WEIGHTS_PATH = os.environ.get("WEIGHTS")


def generate_prompt(example):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:"
    )


@torch.inference_mode()
def _generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_returned_tokens: int,
    max_seq_length: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    if idx.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # advance
        input_pos = input_pos[-1:] + 1

        if idx.device.type == "xla":
            xm.mark_step()

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos]  # include the EOS token

    return idx


class LLMInference:
    INSTRUCTION_TUNED: bool = False

    def __init__(
        self,
        checkpoint_dir: Path = Path(f"checkpoints/tiiuae/falcon-7b"),
        quantize: Literal["llm.int8", "gptq.int4"] = None,
        accelerator: str = "auto",
        strategy: str = "auto",
        devices: int = 1,
        precision: str = "bf16-true",
        adapter_path: Optional[Path] = None,
    ) -> None:
        self.quantize = quantize

        checkpoint_dir = Path(checkpoint_dir)

        if strategy == "fsdp":
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy, transformer_layer_cls={Block}
            )
            strategy = FSDPStrategy(
                auto_wrap_policy=auto_wrap_policy, cpu_offload=False
            )
        self.fabric = fabric = L.Fabric(
            devices=devices,
            precision=precision,
            strategy=strategy,
            accelerator=accelerator,
        )
        fabric.launch()

        check_valid_checkpoint_dir(checkpoint_dir)

        with open(checkpoint_dir / "lit_config.json") as fp:
            self.config = config = Config(**json.load(fp))

        if quantize is not None and devices > 1:
            raise NotImplementedError
        if quantize == "gptq.int4":
            model_file = "lit_model_gptq.4bit.pth"
            if not (checkpoint_dir / model_file).is_file():
                raise ValueError("Please run `python quantize/gptq.py` first")
        else:
            model_file = "lit_model.pth"
        checkpoint_path = checkpoint_dir / model_file

        if adapter_path:
            model = self.load_adapter_model(
                checkpoint_path=checkpoint_path, adapter_path=adapter_path
            )

        else:
            model = self.load_model(checkpoint_path=checkpoint_path)

        model.eval()
        self.model = model = fabric.setup_module(model)
        self.tokenizer = Tokenizer(checkpoint_dir)

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        top_k: int = 200,
        temperature: float = 0.1,
        eos_id=None,
    ) -> str:
        tokenizer = self.tokenizer
        model = self.model
        fabric = self.fabric

        encoded = tokenizer.encode(prompt, device=model.device)
        prompt_length = encoded.size(0)
        max_returned_tokens = prompt_length + max_new_tokens

        t0 = time.perf_counter()
        y = _generate(
            model,
            encoded,
            max_returned_tokens,
            max_seq_length=max_returned_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_id=eos_id,
        )
        t = time.perf_counter() - t0

        model.reset_cache()
        output = tokenizer.decode(y)
        tokens_generated = y.size(0) - prompt_length
        fabric.print(
            f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec",
            file=sys.stderr,
        )
        if fabric.device.type == "cuda":
            fabric.print(
                f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB",
                file=sys.stderr,
            )

        return output

    def instruction_predict(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        top_k: int = 200,
        temperature: float = 0.1,
    ) -> str:
        sample = {"instruction": prompt, "input": input}
        prompt = generate_prompt(sample)
        output = self.__call__(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            temperature=temperature,
            eos_id=self.tokenizer.eos_id,
        )
        output = output.split("### Response:")[1].strip()
        return output

    def eval(self):
        self.model.eval()

    def load_model(self, checkpoint_path: Union[Path, str]):
        fabric = self.fabric
        quantize = self.quantize
        config = self.config

        fabric.print(
            f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
            file=sys.stderr,
        )
        t0 = time.time()
        with fabric.init_module(empty_init=True), quantization(quantize):
            model = GPT(config)
        fabric.print(
            f"Time to instantiate model: {time.time() - t0:.02f} seconds.",
            file=sys.stderr,
        )

        t0 = time.time()
        with lazy_load(checkpoint_path) as checkpoint:
            model.load_state_dict(
                checkpoint.get("model", checkpoint), strict=quantize is None
            )
        fabric.print(
            f"Time to load the model weights: {time.time() - t0:.02f} seconds.",
            file=sys.stderr,
        )
        return model

    def load_lora_model(self, checkpoint_path, lora_path: str):
        return self.load_model(checkpoint_path)

    def load_adapter_model(self, checkpoint_path, adapter_path: str):
        fabric = self.fabric
        quantize = self.quantize
        config = self.config

        fabric.print(
            f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
            file=sys.stderr,
        )
        t0 = time.time()
        with fabric.init_module(empty_init=True), quantization(quantize):
            model = GPT(config)
            add_adapter_v2_parameters_to_linear_layers(model)
        fabric.print(
            f"Time to instantiate model: {time.time() - t0:.02f} seconds.",
            file=sys.stderr,
        )

        t0 = time.time()
        with lazy_load(checkpoint_path) as checkpoint, lazy_load(
            adapter_path
        ) as adapter_checkpoint:
            checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
            model.load_state_dict(checkpoint, strict=quantize is None)
        fabric.print(
            f"Time to load the model weights: {time.time() - t0:.02f} seconds.",
            file=sys.stderr,
        )
        return model
