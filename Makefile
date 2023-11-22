.PHONY: all check test

all: check test

check:
	mypy -p silicon

test:
	llvm-lit test -sv
