# PY_SOURCE_FILES=mosec tests examples setup.py
PY_SOURCE_FILES=llmspec

format:
	@autoflake --in-place --recursive ${PY_SOURCE_FILES}
	@isort --project=mosec ${PY_SOURCE_FILES}
	@black ${PY_SOURCE_FILES}
	# @cargo +nightly fmt --all

lint:
	@pip install -q -e .
	isort --check --diff --project=mosec ${PY_SOURCE_FILES}
	black --check --diff ${PY_SOURCE_FILES}
	pylint -j 8 --recursive=y mosec
	pylint -j 8 --recursive=y --disable=import-error examples --generated-members=numpy.*,torch.*,cv2.*,cv.*
	pydocstyle mosec
	@-rm mosec/_version.py
	pyright --stats
	mypy --non-interactive --install-types ${PY_SOURCE_FILES}
	# cargo +nightly fmt -- --check