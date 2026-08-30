#!/usr/bin/env python3
"""Run training and inference trials for all 10 models."""

import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from inference import clear_predictor_cache, get_predictor
from model_configs import MODEL_CONFIGS, list_models
from trainer import train_model

RESULTS_DIR = Path(__file__).parent / "trial_results"
RESULTS_DIR.mkdir(exist_ok=True)

# Sample inputs for post-training inference smoke tests
TEST_INPUTS = {
    "yelp_5star": ["Great food and amazing service!"],
    "imdb": ["This movie was absolutely fantastic!"],
    "ag_news": ["The stock market reached new highs today."],
    "amazon_polarity": ["Great product", "Works perfectly, highly recommend!"],
    "emotion": ["I am so happy today!"],
    "sms_spam": ["CONGRATULATIONS! You've won $1000! Click here now!"],
    "yelp_polarity": ["Best restaurant in town, incredible food!"],
    "sst2": ["A visually stunning and emotionally powerful film."],
    "dbpedia": ["Apple Inc. is an American multinational technology company."],
    "yahoo_answers": ["How do I fix my computer?", "My laptop keeps crashing."],
}

# Train smallest models first for faster feedback
TRAIN_ORDER = [
    "sms_spam",
    "imdb",
    "emotion",
    "sst2",
    "amazon_polarity",
    "yelp_polarity",
    "ag_news",
    "yahoo_answers",
    "dbpedia",
]


def log(msg: str, log_file):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def test_inference(model_id: str, log_file) -> dict:
    config = MODEL_CONFIGS[model_id]
    texts = TEST_INPUTS[model_id]

    clear_predictor_cache()
    predictor = get_predictor(model_id)
    result = predictor.predict(*texts)

    log(
        f"  Inference OK — {config.name}: "
        f"{result['label']} ({result['confidence']*100:.1f}%)",
        log_file,
    )
    return result


def run_trial(model_id: str, log_file, skip_training: bool = False) -> dict:
    config = MODEL_CONFIGS[model_id]
    if not skip_training and config.is_trained:
        skip_training = True
        log(f"  Model already trained at {config.checkpoint_dir}", log_file)
    trial = {
        "model_id": model_id,
        "name": config.name,
        "started_at": datetime.now().isoformat(),
        "training_skipped": skip_training,
        "training_ok": None,
        "inference_ok": None,
        "metrics": None,
        "prediction": None,
        "error": None,
        "duration_seconds": None,
    }

    start = time.time()
    log(f"\n{'='*60}", log_file)
    log(f"TRIAL: {config.name} ({model_id})", log_file)
    log(f"{'='*60}", log_file)

    try:
        if skip_training:
            if not config.is_trained:
                raise RuntimeError("Model not trained and training skipped")
            log("  Skipping training (already trained)", log_file)
            trial["training_ok"] = True
        else:
            log("  Starting training...", log_file)
            metrics = train_model(model_id)
            trial["metrics"] = metrics
            trial["training_ok"] = True
            log(
                f"  Training OK — test accuracy: "
                f"{metrics['test_accuracy']*100:.1f}%",
                log_file,
            )

        result = test_inference(model_id, log_file)
        trial["prediction"] = {
            "label": result["label"],
            "confidence": result["confidence"],
        }
        trial["inference_ok"] = True

    except Exception as e:
        trial["error"] = str(e)
        trial["training_ok"] = trial["training_ok"] or False
        trial["inference_ok"] = False
        log(f"  FAILED: {e}", log_file)
        traceback.print_exc()

    trial["duration_seconds"] = round(time.time() - start, 1)
    trial["finished_at"] = datetime.now().isoformat()
    log(f"  Duration: {trial['duration_seconds']}s", log_file)
    return trial


def main():
    log_path = RESULTS_DIR / f"trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    results_path = RESULTS_DIR / "latest_results.json"

    with open(log_path, "w") as log_file:
        log("TrainLLM — Full Trial Run", log_file)
        log(f"Models: {len(MODEL_CONFIGS)}", log_file)

        all_results = []

        # Yelp already trained — inference only
        all_results.append(
            run_trial("yelp_5star", log_file, skip_training=True)
        )

        for model_id in TRAIN_ORDER:
            all_results.append(run_trial(model_id, log_file))

        passed = sum(
            1 for r in all_results
            if r["training_ok"] and r["inference_ok"]
        )
        failed = len(all_results) - passed

        log(f"\n{'='*60}", log_file)
        log(f"SUMMARY: {passed}/{len(all_results)} passed, {failed} failed", log_file)
        log(f"{'='*60}", log_file)

        for r in all_results:
            status = "PASS" if r["training_ok"] and r["inference_ok"] else "FAIL"
            acc = ""
            if r.get("metrics"):
                acc = f" (acc={r['metrics']['test_accuracy']*100:.1f}%)"
            elif r["training_skipped"]:
                acc = " (pre-trained)"
            log(f"  [{status}] {r['name']}{acc}", log_file)

        summary = {
            "run_at": datetime.now().isoformat(),
            "passed": passed,
            "failed": failed,
            "total": len(all_results),
            "trials": all_results,
        }

        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)

        log(f"\nResults saved to {results_path}", log_file)
        log(f"Log saved to {log_path}", log_file)

        return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
