# Makefile for Affine Model Training

.PHONY: help install setup test clean train evaluate deploy all

help:
	@echo "Affine Model Training - Available Commands:"
	@echo ""
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Setup environment and verify configuration"
	@echo "  make test        - Run setup verification tests"
	@echo "  make train       - Run complete training pipeline"
	@echo "  make collect     - Collect training data"
	@echo "  make sft         - Run supervised fine-tuning"
	@echo "  make rl          - Run RL training"
	@echo "  make evaluate    - Evaluate trained model"
	@echo "  make deploy      - Deploy to HuggingFace Hub"
	@echo "  make clean       - Clean generated files"
	@echo "  make all         - Install, setup, and train"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "✓ Installation complete"

setup: install
	@echo "Setting up environment..."
	mkdir -p data_cache/affine data_cache/agentgym
	mkdir -p checkpoints models logs
	@echo "✓ Setup complete"

test:
	@echo "Running setup verification..."
	python test_setup.py

collect:
	@echo "Collecting training data..."
	python scripts/1_collect_data.py

sft:
	@echo "Running supervised fine-tuning..."
	python scripts/2_train_sft.py

rl:
	@echo "Running RL training..."
	python scripts/3_train_rl.py

evaluate:
	@echo "Evaluating model..."
	python scripts/4_evaluate.py

deploy:
	@echo "Deploying to HuggingFace Hub..."
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "Error: HF_TOKEN not set"; \
		echo "Set it with: export HF_TOKEN=your_token"; \
		exit 1; \
	fi
	python scripts/5_deploy.py

train:
	@echo "Running complete training pipeline..."
	python scripts/run_full_pipeline.py

clean:
	@echo "Cleaning generated files..."
	rm -rf data_cache/*
	rm -rf checkpoints/*
	rm -rf logs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Clean complete"

all: setup test train
	@echo "✓ Complete pipeline finished"
