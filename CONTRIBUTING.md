# Contributing

We welcome contributions! Here's how to get started.

## Setup

```bash
git clone https://github.com/liodon-ai/juryeval.git
cd juryeval
pip install -e ".[all,judge,lmeval]"
```

## Development

- Run tests: `pytest tests/ -v`
- Add tests for new functionality. We maintain both unit tests and integration tests.
- Follow the existing code style — no docstring or comment boilerplate required.
- All framework integrations (`lmeval/`, etc.) must be guarded by `try/except ImportError`.

## Pull Requests

1. Fork the repo and create a branch from `main`.
2. If adding a feature, include tests.
3. Ensure all tests pass.
4. Open a PR against `liodon-ai/juryeval:main`.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
