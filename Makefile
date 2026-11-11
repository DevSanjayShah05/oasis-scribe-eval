PY := .venv/bin/python
export PYTHONPATH := src:.

setup:            ## create venv and install pinned deps
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

reference:        ## rebuild data/reference from raw CMS files (needs poppler)
	./scripts/fetch_cms.sh && $(PY) scripts/build_reference.py

data:             ## regenerate the SYNTHETIC patients + transcripts
	$(PY) -m scribe_eval.cli generate

baseline:
	$(PY) -m scribe_eval.cli extract --system keyword

llm:              ## ~4 h on an M2 with qwen2.5:7b (resumable)
	./scripts/run_llm_evals.sh

evaluate:
	$(PY) -m scribe_eval.cli evaluate

test:
	$(PY) -m pytest -q

api:
	.venv/bin/uvicorn api.main:app --reload

dashboard:
	.venv/bin/streamlit run dashboard/app.py

.PHONY: setup reference data baseline llm evaluate test api dashboard
