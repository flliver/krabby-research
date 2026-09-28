PYTHON ?= python3

GENERATED := generated

.PHONY: render
render:
	$(PYTHON) schematic/render.py $(GENERATED)
	$(PYTHON) schematic/validate.py $(GENERATED)
