#!/usr/bin/env python3
"""CLI to fine-tune any of the 10 models."""

import argparse

from model_configs import MODEL_CONFIGS, list_models
from trainer import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a text classification model"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model ID to train",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit",
    )
    args = parser.parse_args()

    if args.list or not args.model:
        print("\nAvailable models:\n")
        for config in list_models():
            status = "✓ trained" if config.is_trained else "○ not trained"
            print(f"  {config.id:20s}  {config.name:25s}  [{status}]")
        print()
        return

    train_model(args.model)


if __name__ == "__main__":
    main()
