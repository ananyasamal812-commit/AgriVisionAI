# Broader field-photo research candidate

The live checkpoint still has 38 PlantVillage classes. Accepting an uploaded image
does not mean the plant or condition is recognized correctly. No model in this
project has been validated for every crop, internet photo, or farmer's field.

## Prepared data

Source: [PlantWild v2](https://huggingface.co/datasets/uqtwei2/PlantWild),
[authors' project](https://tqwei05.github.io/PlantWild/).
The source has a CC BY-NC-ND 4.0 license. Keep this as a research candidate;
review redistribution and commercial-use rights before publishing data or weights.
The deployment exporter deliberately excludes this dataset and its candidate.

`datasets/processed/plantwild_field_v1` contains 114 condition classes,
7,673 training images, 1,642 validation images, and 1,642 test images.
Coffee black rot was excluded because only six usable images remained.
Preparation removes exact duplicates and conflicting duplicate labels, verifies
image files, and uses deterministic per-class 70/15/15 splits (rounded).
`preparation.json` records exclusions and `manifest.json` records image hashes.
These random splits do not eliminate near duplicates or shared image sources.

## Reproduce preparation

Download `plantwild_v2.zip` from the source into `datasets/external`, then:

```powershell
.\venv\Scripts\python.exe prepare_field_dataset.py --output datasets/processed/plantwild_field_v1
.\venv\Scripts\python.exe model_workflow.py audit --data datasets/processed/plantwild_field_v1 --classes datasets/processed/plantwild_field_v1/classes.json --output outputs/field-audit
```

Preparation requires a new output directory and never overwrites an existing split.

## Train and evaluate

First validate the pipeline with a small smoke run:

```powershell
.\venv\Scripts\python.exe model_workflow.py train --data datasets/processed/plantwild_field_v1 --classes datasets/processed/plantwild_field_v1/classes.json --audit-report outputs/field-audit/data-audit.json --transfer-backbone --freeze-backbone --max-per-class 2 --epochs 1 --batch-size 2 --output outputs/field-smoke
```

This uses only two images per class per split. Its checkpoint is explicitly marked
`smoke_test_only`; it is not evidence of useful disease recognition.
The completed local smoke run used 228 training and 227 validation images and
scored 3.96% validation accuracy. This is a pipeline check only and is unsuitable
for deployment. The serving API explicitly rejects smoke-test checkpoints.

For a full candidate, omit the sample limit and fine-tune the backbone:

```powershell
.\venv\Scripts\python.exe model_workflow.py train --data datasets/processed/plantwild_field_v1 --classes datasets/processed/plantwild_field_v1/classes.json --audit-report outputs/field-audit/data-audit.json --transfer-backbone --epochs 10 --batch-size 4 --output outputs/field-candidate
.\venv\Scripts\python.exe model_workflow.py evaluate --data datasets/processed/plantwild_field_v1 --classes datasets/processed/plantwild_field_v1/classes.json --audit-report outputs/field-audit/data-audit.json --checkpoint outputs/field-candidate/candidate.pth --output outputs/field-test --split test
```

A CUDA GPU is strongly preferable for the full run. Timing depends on hardware;
the workflow prints progress and writes `progress.json` during training. Validation
selects checkpoints; the test split is not used for selection. Record per-class
metrics, not just overall accuracy. A test result on this dataset does not establish
farmer-photo accuracy.

## Before replacing the serving model

- Measure performance on separately collected, expert-labeled field photos.
- Check near-duplicate/source overlap and class imbalance.
- Test unsupported plants and non-leaf photos; confidence alone is not reliable rejection.
- Review Hindi names and care guidance for the new classes with a domain expert.
- Package the checkpoint with its exact class mapping and matching input size;
  update the displayed supported crops to match the validated model.

Do not overwrite `checkpoints/best_model.pth` with a smoke run. The current exporter
continues to package the existing working model until a candidate is approved by evaluation.
