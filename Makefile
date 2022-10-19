
.PHONY: setup
setup:
#	${MAKE} clean
	./setup.sh
	poetry install
	poetry run pip install -U pip
	cd external/chamfer && poetry run pip install -e .
	cd external/neural_renderer && poetry run pip install -e .

.PHONY: clean
clean:
	rm -rf .wheel
