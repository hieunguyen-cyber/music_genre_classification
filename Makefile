.PHONY: help preprocess feature train evaluate visualize full clean

PY?=python
RUN_NAME?=default

help:
	@echo "Targets:"
	@echo "  preprocess  - scan raw audio -> data/processed/metadata.csv"
	@echo "  feature     - build splits + cached features -> data/features/*.npz"
	@echo "  train       - train model -> outputs/$(RUN_NAME)/checkpoints/best.pt"
	@echo "  evaluate    - evaluate model -> outputs/$(RUN_NAME)/reports/*"
	@echo "  visualize   - generate plots -> outputs/$(RUN_NAME)/figures/*"
	@echo "  full        - run all stages"
	@echo ""
	@echo "Vars:"
	@echo "  RUN_NAME=default"

preprocess:
	$(PY) src/main.py --stage preprocess --run-name $(RUN_NAME)

feature:
	$(PY) src/main.py --stage feature --run-name $(RUN_NAME)

train:
	$(PY) src/main.py --stage train --run-name $(RUN_NAME)

evaluate:
	$(PY) src/main.py --stage evaluate --run-name $(RUN_NAME)

visualize:
	$(PY) src/main.py --stage visualize --run-name $(RUN_NAME)

full:
	$(PY) src/main.py --stage full --run-name $(RUN_NAME)

clean:
	rm -rf outputs/*
