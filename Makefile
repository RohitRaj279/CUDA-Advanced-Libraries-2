# Define compiler
PYTHON = python3

# Define targets
all: run

run:
	$(PYTHON) src/edge_detection.py

clean:
	rm -rf data/output/*

.PHONY: all run clean
