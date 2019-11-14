VENV=${HOME}/.stacksearch
PYTHON=python3

virtual-env:
	$(PYTHON) -m venv $(VENV)

base-deps:
	. $(VENV)/bin/activate && \
	pip install -r requirements.txt

setup: virtual-env base-deps

clean:
	rm -rf $(VENV)