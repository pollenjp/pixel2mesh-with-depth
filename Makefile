SHELL := /bin/bash

.PHONY: setup
setup:
	${MAKE} clean
	./setup.sh
	rm -rf .venv
	poetry install

.PHONY: clean
clean:
	rm -rf ".venv"
