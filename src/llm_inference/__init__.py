"""Inference API for LLaMA"""

from .download import prepare_weights
from .model import LLMInference
from .serve import ServeLLaMA

__version__ = "0.0.5dev0"
