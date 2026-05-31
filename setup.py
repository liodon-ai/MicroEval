from setuptools import setup, find_packages

setup(
    name="juryeval",
    version="0.4.0",
    description="Lightweight NLP/LLM evaluation toolkit — metrics, judges, significance testing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Liodon AI",
    author_email="info@liodon.ai",
    url="https://github.com/liodon-ai/juryeval",
    license="MIT",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "juryeval=juryeval.cli:main",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "nltk",
    ],
    extras_require={
        "full": [
            "scikit-learn",
            "sacrebleu",
            "rouge-score",
            "transformers",
            "torch",
            "sentence-transformers",
        ],
        "judge": [
            "openai>=1.0.0",
        ],
        "semantic": [
            "sentence-transformers",
        ],
        "lmeval": [
            "lm-eval>=0.4.0",
        ],
        "all": [
            "scikit-learn",
            "sacrebleu",
            "rouge-score",
            "transformers",
            "torch",
            "sentence-transformers",
            "openai>=1.0.0",
            "lm-eval>=0.4.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
