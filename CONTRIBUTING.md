# Contributing

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Reporting Bugs

When submitting a new bug report, please first [search](https://github.com/ew2664/mabby/issues) for an existing or similar report. If you believe you've come across a unique problem, then use one of our existing [issue templates](https://github.com/ew2664/mabby/issues/new/choose). Duplicate issues or issues that don't use one of our templates may get closed without a response.

## Development

Before contributing, make sure you have [poetry](https://python-poetry.org/) installed.

1. Fork the [repository](https://github.com/ew2664/mabby) on GitHub.

1. Clone the repository from your GitHub.

1. Setup development environment (`make install`).

1. Setup [pre-commit](https://pre-commit.com/) hooks (`poetry run pre-commit install`).

1. Check out a new branch and make your modifications.

1. Add test cases for all your changes.

1. Run `make lint` and `make test` and ensure they pass.

   > Run `make format` to fix the linting errors that are auto-fixable.

   > Run `make coverage` to run unit tests only and generate an HTML coverage report.

1. Commit your changes following our [commit conventions](#commit-conventions).

1. Push your changes to your fork of the repository.

1. Open a [pull request](https://github.com/ew2664/mabby/pulls)! 🎉🎉🎉

## Commit Conventions

We follow [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/). When opening a pull request, please be sure that both the pull request title and each commit in the pull request has one of the following prefixes:

| Prefix      | Description                                                               | SemVer  |
| :---------- | ------------------------------------------------------------------------- | :-----: |
| `feat:`     | a new feature                                                             | `MINOR` |
| `fix:`      | a bug fix                                                                 | `PATCH` |
| `refactor:` | a code change that neither fixes a bug nor adds a new feature             | `PATCH` |
| `docs:`     | a documentation-only change                                               | `PATCH` |
| `chore:`    | any other change that does not affect the published module (e.g. testing) |  none   |
