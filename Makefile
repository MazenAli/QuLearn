# Minimal makefile for Sphinx documentation and project maintenance tasks
#

# Variables that can be set from the command line or environment
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = build

# Default target executed when no arguments are given to make.
default: all

all: docs-html format format_check static test_coverage

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docs-%:
	@$(SPHINXBUILD) -M $* "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

format:
	black --line-length 100 qulearn tests
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 qulearn tests

format_check:
	black --line-length 100 --check qulearn tests
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 qulearn tests --check-only

static:
	flake8 qulearn tests
	mypy qulearn tests --ignore-missing-imports --no-strict-optional

# Testing
test:
	pytest tests/

test_coverage:
	coverage run --source=qulearn --module pytest -v tests/ && coverage report -m

.PHONY: help docs-% format format_check static test test_coverage
