.PHONY: pre-commit
pre-commit:
	poetry run pre-commit run --all-files
