#!/usr/bin/env python3
"""Interactive Gradio UI for testing and fine-tuning all 10 models."""

import json
import threading
from pathlib import Path

import gradio as gr
import pandas as pd

from inference import clear_predictor_cache, get_predictor
from model_configs import MODEL_CONFIGS, list_models
from trainer import train_model

# Example inputs per model for quick testing
EXAMPLES = {
    "yelp_5star": [
        ["Amazing food and wonderful service! Will definitely come back."],
        ["Terrible experience. Food was cold and the waiter was rude."],
        ["It was okay, nothing special but not bad either."],
    ],
    "imdb": [
        ["This movie was absolutely fantastic! A masterpiece of cinema."],
        ["Boring plot, bad acting, waste of two hours."],
    ],
    "ag_news": [
        ["The stock market reached new highs today as tech companies rallied."],
        ["The national team won the championship in a thrilling final match."],
    ],
    "amazon_polarity": [
        ["Great product", "Works perfectly, fast shipping, highly recommend!"],
        ["Disappointed", "Broke after one week. Complete waste of money."],
    ],
    "emotion": [
        ["I am so happy today, everything is going wonderfully!"],
        ["I feel so lonely and sad, nothing seems to help."],
    ],
    "sms_spam": [
        ["Hey, are we still meeting for lunch tomorrow?"],
        ["CONGRATULATIONS! You've won $1000! Click here to claim now!"],
    ],
    "yelp_polarity": [
        ["Best restaurant in town! Incredible flavors and friendly staff."],
        ["Worst dining experience ever. Never going back."],
    ],
    "sst2": [
        ["A visually stunning and emotionally powerful film."],
        ["Dull and predictable with no redeeming qualities."],
    ],
    "dbpedia": [
        ["Apple Inc. is an American multinational technology company headquartered in Cupertino, California."],
    ],
    "yahoo_answers": [
        ["How do I fix my computer?", "My laptop keeps crashing when I open Chrome. What should I do?"],
    ],
}


def get_model_choices():
    return [(f"{c.name} ({'trained' if c.is_trained else 'not trained'})", c.id) for c in list_models()]


def get_model_status(model_id: str) -> str:
    config = MODEL_CONFIGS[model_id]
    if config.is_trained:
        metrics_path = config.checkpoint_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            acc = metrics.get("test_accuracy", 0) * 100
            return f"✅ **{config.name}** is trained (test accuracy: {acc:.1f}%)"
        return f"✅ **{config.name}** is trained"
    return f"⚠️ **{config.name}** has not been trained yet. Use the Fine-Tune tab to train it."


def predict(model_id: str, *texts):
    if not model_id:
        return "Select a model first.", None

    config = MODEL_CONFIGS[model_id]
    if not config.is_trained:
        return get_model_status(model_id), None

    try:
        clear_predictor_cache()
        predictor = get_predictor(model_id)
        result = predictor.predict(*texts)

        if result.get("error"):
            return result["error"], None

        label = result["label"]
        confidence = result["confidence"] * 100
        probs = result["probabilities"]

        summary = (
            f"### Prediction: **{label}**\n\n"
            f"Confidence: **{confidence:.1f}%**"
        )

        df = pd.DataFrame(
            {"Label": list(probs.keys()), "Probability": [f"{v*100:.1f}%" for v in probs.values()]}
        )
        df = df.sort_values("Probability", ascending=False).reset_index(drop=True)

        return summary, df

    except Exception as e:
        return f"Error: {e}", None


def update_inputs(model_id: str):
    if not model_id:
        return [gr.update(visible=False)] * 2 + [gr.update(value="Select a model.")]

    config = MODEL_CONFIGS[model_id]
    status = get_model_status(model_id)

    updates = []
    for i in range(2):
        if i < len(config.text_fields):
            updates.append(
                gr.update(
                    visible=True,
                    label=config.text_field_labels[i],
                    placeholder=f"Enter {config.text_field_labels[i].lower()}...",
                    value="",
                )
            )
        else:
            updates.append(gr.update(visible=False, value=""))

    updates.append(gr.update(value=status))
    return updates


def load_example(model_id: str):
    examples = EXAMPLES.get(model_id, [])
    if not examples:
        return "", ""
    ex = examples[0]
    if len(ex) == 1:
        return ex[0], ""
    return ex[0], ex[1]


def run_training(model_id: str, progress=gr.Progress()):
    if not model_id:
        yield "Select a model to train."
        return

    config = MODEL_CONFIGS[model_id]
    log_lines = [f"Starting fine-tuning for **{config.name}**...\n"]
    yield "\n".join(log_lines)

    done = threading.Event()
    error_holder = [None]
    metrics_holder = [None]

    def progress_callback(msg: str):
        log_lines.append(msg)

    def train_thread():
        try:
            metrics_holder[0] = train_model(model_id, progress_callback=progress_callback)
        except Exception as e:
            error_holder[0] = str(e)
        finally:
            done.set()

    thread = threading.Thread(target=train_thread)
    thread.start()

    while not done.is_set():
        yield "\n".join(log_lines[-50:])
        done.wait(timeout=2)

    if error_holder[0]:
        log_lines.append(f"\n❌ Training failed: {error_holder[0]}")
    else:
        clear_predictor_cache()
        acc = metrics_holder[0]["test_accuracy"] * 100
        log_lines.append(f"\n✅ Training complete! Test accuracy: {acc:.1f}%")

    yield "\n".join(log_lines)


def build_ui():
    with gr.Blocks(
        title="TrainLLM — Model Testing & Fine-Tuning",
    ) as app:
        gr.Markdown(
            """
            # 🧠 TrainLLM — Interactive Model Hub

            Test fine-tuned text classification models or train new ones.
            Select a model below to get started.
            """
        )

        model_dropdown = gr.Dropdown(
            choices=get_model_choices(),
            label="Select Model",
            value="yelp_5star",
            interactive=True,
        )

        with gr.Tabs():
            with gr.Tab("🔮 Test Model"):
                status_md = gr.Markdown(get_model_status("yelp_5star"))

                with gr.Row():
                    text_input_1 = gr.Textbox(
                        label="Text",
                        placeholder="Enter text to classify...",
                        lines=4,
                        visible=True,
                    )
                    text_input_2 = gr.Textbox(
                        label="Additional Text",
                        placeholder="Enter additional text...",
                        lines=4,
                        visible=False,
                    )

                with gr.Row():
                    predict_btn = gr.Button("Predict", variant="primary", scale=2)
                    example_btn = gr.Button("Load Example", scale=1)
                    clear_btn = gr.Button("Clear", scale=1)

                result_md = gr.Markdown()
                prob_table = gr.Dataframe(
                    headers=["Label", "Probability"],
                    label="Class Probabilities",
                    interactive=False,
                )

                gr.Markdown("*Tip: switch models to see task-specific example inputs in the Fine-Tune tab.*")

                def do_predict(model_id, t1, t2):
                    config = MODEL_CONFIGS[model_id]
                    texts = [t1]
                    if len(config.text_fields) > 1:
                        texts.append(t2)
                    return predict(model_id, *texts)

                predict_btn.click(
                    do_predict,
                    inputs=[model_dropdown, text_input_1, text_input_2],
                    outputs=[result_md, prob_table],
                )
                example_btn.click(
                    load_example,
                    inputs=[model_dropdown],
                    outputs=[text_input_1, text_input_2],
                )
                clear_btn.click(
                    lambda: ("", "", None, None),
                    outputs=[text_input_1, text_input_2, result_md, prob_table],
                )

            with gr.Tab("🏋️ Fine-Tune"):
                gr.Markdown(
                    """
                    Fine-tune a model using the same training pipeline as `main.py`:
                    train/val/test splits, learning rate scheduling, early stopping,
                    and checkpoint saving.
                    """
                )

                train_status = gr.Markdown(get_model_status("yelp_5star"))
                train_btn = gr.Button("Start Fine-Tuning", variant="primary")
                train_log = gr.Markdown("Training output will appear here.")

                train_btn.click(
                    run_training,
                    inputs=[model_dropdown],
                    outputs=[train_log],
                ).then(
                    lambda mid: get_model_status(mid),
                    inputs=[model_dropdown],
                    outputs=[train_status],
                )

            with gr.Tab("📋 Model Info"):
                info_md = gr.Markdown()

                def show_info(model_id):
                    c = MODEL_CONFIGS[model_id]
                    trained = "Yes ✅" if c.is_trained else "No ⚠️"
                    fields = ", ".join(
                        f"`{f}`" for f in c.text_fields
                    )
                    return f"""
### {c.name}

{c.description}

| Setting | Value |
|---------|-------|
| Model ID | `{c.id}` |
| Base Model | `{c.model_name}` |
| Dataset | `{c.dataset}` |
| Labels | {c.num_labels} ({', '.join(c.label_names)}) |
| Text Fields | {fields} |
| Train / Val / Test | {c.train_size} / {c.eval_size} / {c.test_size} |
| Batch Size | {c.batch_size} |
| Learning Rate | {c.learning_rate} |
| Epochs | {c.epochs} |
| Max Length | {c.max_length} |
| Trained | {trained} |
| Checkpoint | `{c.checkpoint_dir}` |
"""

                info_md.value = show_info("yelp_5star")

        def on_model_change(model_id):
            return [
                *update_inputs(model_id),
                show_info(model_id),
            ]

        model_dropdown.change(
            on_model_change,
            inputs=[model_dropdown],
            outputs=[text_input_1, text_input_2, status_md, info_md],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
