NAME := mabby
INSTALL_STAMP := .install.stamp
POETRY := $(shell command -v poetry 2> /dev/null)

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo ""
	@echo "  install      install packages and prepare environment"
	@echo "  clean        remove all temporary files"
	@echo "  deep-clean   remove all untracked files"
	@echo "  lint         run code linters"
	@echo "  format       reformat code"
	@echo "  test         run all tests"
	@echo "  unit         run unit tests"
	@echo "  integration  run integration tests"
	@echo "  coverage     run unit tests and generate coverage report"
	@echo "  pre-commit   run all pre-commit hooks"
	@echo ""
	@echo "Check the Makefile to see what exactly each target is doing."

.PHONY: install
install: $(INSTALL_STAMP)
$(INSTALL_STAMP): pyproject.toml poetry.lock
	@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
	$(POETRY) install
	touch $(INSTALL_STAMP)

.PHONY: deep-clean
deep-clean:
	git clean -fdx

.PHONY: clean
clean:
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf $(INSTALL_STAMP) .coverage coverage cover htmlcov logs build dist *.egg-info .mypy_cache .pytest_cache .ruff_cache

.PHONY: lint
lint: $(INSTALL_STAMP)
	$(POETRY) run ruff check ./tests/ $(NAME) --exit-zero
	$(POETRY) run black --check ./tests/ $(NAME) --diff
	$(POETRY) run mdformat --check $(NAME)
	$(POETRY) run mypy $(NAME) --ignore-missing-imports

.PHONY: format
format: $(INSTALL_STAMP)
	$(POETRY) run ruff check --fix ./tests/ $(NAME) --exit-zero
	$(POETRY) run black ./tests/ $(NAME)
	$(POETRY) run mdformat $(NAME)

.PHONY: test
test: $(INSTALL_STAMP)
	$(POETRY) run pytest ./tests/ --cov-report term-missing --cov-fail-under 80 --cov $(NAME)

.PHONY: unit
unit: $(INSTALL_STAMP)
	$(POETRY) run pytest ./tests/unit/ --cov-report term-missing --cov-fail-under 80 --cov $(NAME)

.PHONY: integration
integration: $(INSTALL_STAMP)
	$(POETRY) run pytest ./tests/integration/ $(NAME)

.PHONY: coverage
coverage: $(INSTALL_STAMP)
	$(POETRY) run pytest ./tests/unit/ --cov-report html --cov-fail-under 80 --cov $(NAME)

.PHONY: pre-commit
pre-commit: $(INSTALL_STAMP)
	$(POETRY) run pre-commit run --all-files
