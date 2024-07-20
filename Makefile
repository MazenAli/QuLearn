# Minimal makefile for Sphinx documentation and project maintenance tasks
#

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = build

default: all

all: docs-html format format_check static test_coverage secrets_check

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

test:
	pytest tests/

test_coverage:
	coverage run --source=qulearn --module pytest -v tests/ && coverage report -m
	coverage xml

secrets_check:
	@git secrets --scan -r

.PHONY: help docs-% format format_check static test test_coverage secrets_check
