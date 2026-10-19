.PHONY: install test lint demo clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

demo:
	hcp-engine run --out outputs

clean:
	rm -rf outputs .pytest_cache .ruff_cache build dist *.egg-info
