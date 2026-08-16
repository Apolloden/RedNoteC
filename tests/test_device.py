from pathlib import Path

import torch

from rednote_aigt.training.train import build_training_arguments_kwargs
from rednote_aigt.utils.device import get_torch_device, resolve_torch_device


def test_device_selection_returns_cpu_when_prefer_mps_false():
    device = get_torch_device(prefer_mps=False, prefer_cuda=False)
    assert device.type == "cpu"


def test_device_selection_does_not_crash_if_mps_missing(monkeypatch):
    class Backends:
        pass

    monkeypatch.setattr(torch, "backends", Backends(), raising=False)
    device = get_torch_device(prefer_mps=True, prefer_cuda=False)
    assert device.type == "cpu"


def test_resolve_cpu_forces_cpu():
    assert resolve_torch_device("cpu").type == "cpu"


def test_training_arguments_builder_drops_unsupported_args(tmp_path):
    config = {
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 1,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "seed": 42,
        "fp16": False,
        "bf16": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "no_cuda": True,
    }
    kwargs = build_training_arguments_kwargs(
        config=config,
        output_dir=tmp_path,
        eval_strategy="steps",
        eval_steps=1,
        save_steps=1,
        max_steps=3,
        selected_device="mps",
    )
    import inspect

    from transformers import TrainingArguments

    supported = inspect.signature(TrainingArguments.__init__).parameters
    assert set(kwargs).issubset(set(supported))
    assert kwargs["per_device_train_batch_size"] == 2
    assert kwargs["dataloader_pin_memory"] is False


def test_train_cli_accepts_device_options(monkeypatch):
    import scripts.train as train_script

    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--model",
            "tfidf_logreg",
            "--device",
            "auto",
            "--batch-size",
            "2",
            "--max-length",
            "128",
        ],
    )
    args = train_script.parse_args()
    assert args.device == "auto"
    assert args.batch_size == 2
    assert args.max_length == 128


def test_evaluate_cli_accepts_cpu_device(monkeypatch):
    import scripts.evaluate as evaluate_script

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--model-dir",
            "models/tfidf_logreg",
            "--device",
            "cpu",
        ],
    )
    args = evaluate_script.parse_args()
    assert args.device == "cpu"
    assert isinstance(args.model_dir, Path)
