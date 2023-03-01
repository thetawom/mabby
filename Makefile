# Thanks to https://blog.mathieu-leplatre.info/tips-for-your-makefile-with-python.html
# Thanks to https://www.thapaliya.com/en/writings/well-documented-makefiles/

NAME := mabby
INSTALL_STAMP := .install.stamp
POETRY := $(shell command -v poetry 2> /dev/null)

.DEFAULT_GOAL := help

.PHONY: help
help:
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Building
.PHONY: install
install: $(INSTALL_STAMP) ## install packages and prepare environment
$(INSTALL_STAMP): pyproject.toml poetry.lock
	@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
	$(POETRY) install
	touch $(INSTALL_STAMP)

##@ Linting
.PHONY: lint
lint: $(INSTALL_STAMP) ## run code linters
	$(POETRY) run ruff check ./tests/ $(NAME) --exit-zero
	$(POETRY) run black --check ./tests/ $(NAME) --diff
	$(POETRY) run mdformat --check $(NAME)
	$(POETRY) run pyproject-fmt --check ./pyproject.toml
	$(POETRY) run mypy $(NAME) --ignore-missing-imports

.PHONY: lints
lints: lint

.PHONY: format
format: $(INSTALL_STAMP) ## reformat code
	$(POETRY) run ruff check --fix ./tests/ $(NAME) --exit-zero
	$(POETRY) run black ./tests/ $(NAME)
	$(POETRY) run mdformat ./tests/ $(NAME)
	$(POETRY) run pyproject-fmt ./pyproject.toml

.PHONY: fix
fix: format

.PHONY: pre-commit
pre-commit: $(INSTALL_STAMP) ## run all pre-commit hooks
	$(POETRY) run pre-commit run --all-files

##@ Testing
.PHONY: test
test: $(INSTALL_STAMP) ## run all tests
	$(POETRY) run pytest ./tests/ --cov-report term-missing --cov $(NAME)

.PHONY: tests
tests: test

.PHONY: unit
unit: $(INSTALL_STAMP) ## run all unit tests
	$(POETRY) run pytest ./tests/unit/ --cov-report lcov --cov $(NAME)

.PHONY: test-unit
test-unit: unit

.PHONY: integration
integration: $(INSTALL_STAMP) ## run all integration tests
	$(POETRY) run pytest ./tests/integration/ $(NAME)

.PHONY: test-integration
test-integration: integration

.PHONY: coverage
coverage: $(INSTALL_STAMP) ## generate HTML coverage report
	$(POETRY) run pytest ./tests/unit/ --cov-report term-missing --cov-report html --cov $(NAME)

##@ Cleanup
.PHONY: deep-clean
deep-clean: ## remove all untracked files
	git clean -fdx

.PHONY: clean
clean: ## remove all temporary files
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf $(INSTALL_STAMP) .coverage coverage *.lcov cover htmlcov logs build dist *.egg-info .mypy_cache .pytest_cache .ruff_cache
