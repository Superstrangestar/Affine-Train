from setuptools import setup, find_packages

setup(
    name="affine-model-training",
    version="1.0.0",
    description="Training pipeline for fine-tuning Affine-0004 model",
    author="Affine Training Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "datasets>=2.14.0",
        "wandb>=0.16.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "affine-train=training.train:main",
            "affine-eval=evaluation.evaluate:main",
            "affine-deploy=deployment.deploy:main",
        ],
    },
)
