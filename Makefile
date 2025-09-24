POETRY ?= poetry
PYTHON_VERSION ?= 3.13
DOCS_SRC ?= docs/source
DOCS_BUILD ?= docs/_build

.PHONY: ci install lint typecheck test docs precommit clean lock envinfo

ci: clean install lint typecheck test docs precommit

install:
	@set -e; \
	PY=$$(command -v python || command -v python3); \
	if [ -z "$$PY" ]; then echo "python or python3 not found in PATH"; exit 1; fi; \
	echo "Using python: $$PY"; \
	$(POETRY) config virtualenvs.create true; \
	$(POETRY) config virtualenvs.in-project true; \
	$(POETRY) config virtualenvs.prefer-active-python true || true; \
	rm -rf .venv || true; \
	$(POETRY) env use $$PY; \
	$(POETRY) lock --no-update || $(POETRY) lock; \
	$(POETRY) env remove --all || true; \
	$(POETRY) install --with dev,docs; \
	$(MAKE) envinfo

lint:
	$(POETRY) run ruff check .

typecheck:
	$(POETRY) run pyright

test:
	$(POETRY) run pytest -q

docs:
	$(POETRY) run sphinx-build -b html $(DOCS_SRC) $(DOCS_BUILD)

precommit:
	$(POETRY) run pre-commit run --all-files --show-diff-on-failure

lock:
	$(POETRY) lock --no-update || $(POETRY) lock

envinfo:
	$(POETRY) env info
	$(POETRY) run python -c "import sys; print(sys.version)"

clean:
	rm -rf .venv $(DOCS_BUILD)


