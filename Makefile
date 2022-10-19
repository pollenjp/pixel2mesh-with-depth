
.PHONY: setup
setup:
#	${MAKE} clean
	./setup.sh

.PHONY: clean
clean:
	rm -rf .wheel
