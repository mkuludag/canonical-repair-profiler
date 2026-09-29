.PHONY: help install test cov lint demo run dash sample figures clean

PY ?= python3

help:
	@echo "Canonical Repair Profiler - make targets"
	@echo "  make install   Install runtime + dev dependencies"
	@echo "  make test      Run the pytest suite (offline, all external calls mocked)"
	@echo "  make cov       Run tests with a coverage report"
	@echo "  make sample    Rebuild the bundled offline sample from local parquet"
	@echo "  make demo      Offline inference demo against the bundled sample (no cloud)"
	@echo "  make run       Run the agent pipeline on one signature (SIG=32768)"
	@echo "  make dash      Launch the Streamlit dashboard"
	@echo "  make figures   Regenerate figures"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest pytest-cov

test:
	$(PY) -m pytest

cov:
	$(PY) -m pytest --cov=src --cov-report=term-missing

sample:
	$(PY) scripts/build_sample.py

SIG ?= 32954
demo:
	CRP_USE_SAMPLE=1 $(PY) cli.py infer --vehicle "F-150" --part 6148 \
		--concern "engine knock on cold start, heavy oil consumption, misfire on cylinders 1 and 4"

run:
	$(PY) cli.py run --signature $(SIG)

dash:
	$(PY) -m streamlit run app.py

figures:
	$(PY) figures/make_figures.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov *.egg-info build dist
