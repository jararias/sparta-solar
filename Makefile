
-include .env
export UV_PUBLISH_TOKEN
export UV_PUBLISH_TEST_TOKEN

.PHONY: ipython3
ipython3:
	@uv run ipython --profile-dir=.ipython

.PHONY: init-venv
init-venv:
	@uv sync --reinstall --all-extras
	@attr -s com.dropbox.ignored -V 1 .venv  # instruct dropbox to ignore the .venv folder

.PHONY: clean
clean:
	@rm -rf build
	@rm -rf dist
	@rm -rf site
	@rm -rf *.egg-info
	@rm -rf .pytest_cache
	@rm -f MANIFEST
	@find . -name "__pycache__" -print0 | xargs -0 -I {} /bin/rm -rf "{}"

.PHONY: tests
coverage:
	@uv run coverage run -m pytest -v --junit-xml=junit.xml tests/unit/
	@uv run coverage xml -o coverage.xml
	@uv run genbadge tests -i junit.xml -o docs/images/tests-badge.svg
	@uv run genbadge coverage -i coverage.xml -o docs/images/coverage-badge.svg
	@echo "Badges generated in docs/images/*-badge.svg"
	@rm -f junit.xml coverage.xml .coverage

.PHONY: docs
docs:
	@echo "Building documentation with Zensical..."
	@uv run zensical build --clean
	@echo "Documentation built successfully in site/ directory"
	@attr -s com.dropbox.ignored -V 1 site  # instruct dropbox to ignore the .venv folder

.PHONY: docs-serve
docs-serve:
	@echo "Serving documentation locally..."
	@uv run zensical serve --open

.PHONY: docs-clean
docs-clean:
	@echo "Cleaning documentation build..."
	@rm -rf site/

.PHONY: publish
publish:
	@uv run scripts/create_env.py
	@test -n "$(UV_PUBLISH_TOKEN)" || (echo "Error: UV_PUBLISH_TOKEN no está definido (créalo en .env)"; exit 1)
	@uv build
	@uv publish
	@rm -f .env

.PHONY: publish-test
publish-test:
	@uv run scripts/create_env.py
	@test -n "$(UV_PUBLISH_TEST_TOKEN)" || (echo "Error: UV_PUBLISH_TEST_TOKEN no está definido (créalo en .env)"; exit 1)
	@uv build
	@uv publish --publish-url https://test.pypi.org/legacy/ --token $(UV_PUBLISH_TEST_TOKEN)
	@rm -f .env
