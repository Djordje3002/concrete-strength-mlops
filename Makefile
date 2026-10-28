.PHONY: install train test check
install:
	python -m pip install -e ".[dev]"
train:
	python -m concrete_ml train
test:
	python -m pytest -q
check:
	python -m ruff check .
	python -m ruff format --check .
