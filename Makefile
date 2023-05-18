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

cspell-collect-unknown-words:
	@# install cspell by `npm i -g cspell`
	@# @keming, please feel free to modify the glob pattern if you think it's too much
	@cspell --words-only --unique '{*.py{,nb},{!({,.}venv)/**/*.{html,py,js,ts,css,md,yaml,yml,json,txt,code-snippets,ipynb,Rmd,R},.github/**/*.{md,yaml,yml}}}' | LC_ALL='C' sort --ignore-case > project-words.txt

.PHONY: *
