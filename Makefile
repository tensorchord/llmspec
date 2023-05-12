PY_SOURCE=llmspec

.DEFAULT_GOAL:=build

build:
	@pdm build

lint:
	@black --check --diff ${PY_SOURCE}
	@ruff check .

format:
	@black ${PY_SOURCE}
	@ruff check --fix .


.PHONY: *