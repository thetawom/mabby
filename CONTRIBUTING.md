# Contributing

[![poetry](https://img.shields.io/badge/packaging-poetry-008adf)](https://python-poetry.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

We welcome and value all types of contributions, from bug reports to feature additions. Please make sure to read the relevant section(s) below before making your contribution.

And if you like the project but just don't have time to contribute, there are other easy ways to support the project and show your appreciation. We'd love if you could:

- Star the project
- Refer this project in your project's README
- Mention the project at local meetups and tell your friends/colleagues

Once again, thank you for supporting the project and taking the time to contribute!

## Code of Conduct

Please read and follow our [Code of Conduct](https://ew2664.github.io/mabby/code_of_conduct/).

## Reporting Bugs

When submitting a new bug report, please first [search](https://github.com/ew2664/mabby/issues) for an existing or similar report. If you believe you've come across a unique problem, then use one of our existing [issue templates](https://github.com/ew2664/mabby/issues/new/choose). Duplicate issues or issues that don't use one of our templates may get closed without a response.

## Development

Before contributing, make sure you have [Python 3.9+](https://www.python.org/downloads/) and [poetry](https://python-poetry.org/) installed.

1. Fork the [repository](https://github.com/ew2664/mabby) on GitHub.

1. Clone the repository from your GitHub.

1. Setup development environment (`make install`).

1. Setup [pre-commit](https://pre-commit.com/) hooks (`poetry run pre-commit install`).

1. Check out a new branch and make your modifications.

1. Add test cases for all your changes.

1. Run `make lint` and `make test` and ensure they pass.

    TIP: Run `make format` to fix the linting errors that are auto-fixable.

    TIP: Run `make coverage` to run unit tests only and generate an HTML coverage report.

1. Commit your changes following our [commit conventions](#commit-conventions).

1. Push your changes to your fork of the repository.

1. Open a [pull request](https://github.com/ew2664/mabby/pulls)! :tada::tada::tada:

## Commit Conventions

We follow [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/). When opening a pull request, please be sure that both the pull request title and each commit in the pull request has one of the following prefixes:

| Prefix      | Description                                                               | SemVer  |
| :---------- | ------------------------------------------------------------------------- | :-----: |
| `feat:`     | a new feature                                                             | `MINOR` |
| `fix:`      | a bug fix                                                                 | `PATCH` |
| `refactor:` | a code change that neither fixes a bug nor adds a new feature             | `PATCH` |
| `docs:`     | a documentation-only change                                               | `PATCH` |
| `chore:`    | any other change that does not affect the published module (e.g. testing) |  none   |
