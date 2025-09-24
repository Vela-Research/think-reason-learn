POETRY ?= poetry
PYTHON_VERSION ?= 3.13
DOCS_SRC ?= docs/source
DOCS_BUILD ?= docs/_build

.PHONY: ci install lint typecheck test docs precommit

ci: install lint typecheck test docs precommit

install:
	$(POETRY) env use $(PYTHON_VERSION)
	$(POETRY) install --with dev,docs

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


