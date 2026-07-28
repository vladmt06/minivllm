from __future__ import annotations

import pytest
import torch

from minivllm.model.loader import load_model, model_path

# Correctness is checked on CPU/fp32 throughout. MPS+fp16 will not reproduce HF
# fp32 bit-for-bit, and chasing that gap teaches nothing about paging.
DEVICE = torch.device("cpu")
DTYPE = torch.float32

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
    "The three laws of robotics state that",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time in a distant galaxy",
    "The mitochondria is the",
    "import numpy as np\n\n# Compute the mean of an array\n",
]


@pytest.fixture(scope="session")
def path():
    return model_path()


@pytest.fixture(scope="session")
def tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path))


@pytest.fixture(scope="session")
def model(path):
    m, _cfg = load_model(path, device=DEVICE, dtype=DTYPE)
    return m


@pytest.fixture(scope="session")
def cfg(path):
    from minivllm.config import ModelConfig

    return ModelConfig.from_hf(path)


@pytest.fixture(scope="session")
def hf_model(path):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(str(path), dtype=DTYPE)
    return m.eval().to(DEVICE)
