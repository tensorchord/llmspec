PY_SOURCE=llmspec tests

.DEFAULT_GOAL:=build

build:
	@pdm build

lint:
	@ruff check ${PY_SOURCE}

format:
	@ruff format ${PY_SOURCE}

test:
	@pytest -vv -s

# install cspell by `npm i -g cspell`
cspell-collect-unknown-words:
	@cspell --words-only --unique '{*.py{,nb},{!({,.}venv)/**/*.{html,py,js,ts,css,md,yaml,yml,json,txt,code-snippets,ipynb,Rmd,R},.github/**/*.{md,yaml,yml}}}' | LC_ALL='C' sort --ignore-case > project-words.txt

.PHONY: *
