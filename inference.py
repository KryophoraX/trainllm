"""Load fine-tuned models and run inference."""

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_configs import ROOT_DIR, ModelConfig, get_model_config


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ModelPredictor:
    def __init__(self, model_id: str):
        self.config = get_model_config(model_id)
        self.device = get_device()
        self.model = None
        self.tokenizer = None

    def is_available(self) -> bool:
        return self.config.is_trained

    def load(self):
        if self.model is not None:
            return

        config = self.config
        save_dir = config.checkpoint_dir

        if save_dir.exists() and (save_dir / "config.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(save_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        elif config.legacy_checkpoint:
            legacy_path = ROOT_DIR / config.legacy_checkpoint
            if not legacy_path.exists():
                raise FileNotFoundError(
                    f"No checkpoint found for {config.name}. "
                    f"Train it first with: python train.py --model {config.id}"
                )
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=config.num_labels,
            )
            state_dict = torch.load(legacy_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                f"No checkpoint found for {config.name}. "
                f"Train it first with: python train.py --model {config.id}"
            )

        self.model.to(self.device)
        self.model.eval()

    def _build_text(self, texts: list[str]) -> str:
        if self.config.preprocess == "concat":
            return " ".join(t for t in texts if t)
        return texts[0] if texts else ""

    def predict(self, *texts: str) -> dict:
        self.load()

        text = self._build_text(list(texts))
        if not text.strip():
            return {
                "error": "Please enter some text to classify.",
                "label": None,
                "confidence": 0.0,
                "probabilities": {},
            }

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        probabilities = {
            self.config.label_names[i]: float(probs[i])
            for i in range(len(self.config.label_names))
        }

        return {
            "label": self.config.label_names[pred_idx],
            "label_index": pred_idx,
            "confidence": confidence,
            "probabilities": probabilities,
            "text": text,
        }

    def unload(self):
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_predictor_cache: dict[str, ModelPredictor] = {}


def get_predictor(model_id: str) -> ModelPredictor:
    if model_id not in _predictor_cache:
        _predictor_cache[model_id] = ModelPredictor(model_id)
    return _predictor_cache[model_id]


def clear_predictor_cache():
    for predictor in _predictor_cache.values():
        predictor.unload()
    _predictor_cache.clear()
