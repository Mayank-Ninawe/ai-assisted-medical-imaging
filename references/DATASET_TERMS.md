# Dataset Access, Terms, and Citation

Status: Phase 12 — pending final classification-dataset decision

## Educational-use notice

This project is for educational purposes only, not for clinical use.
It does not provide clinically validated diagnoses or treatment advice.

## Workflow A — Chest X-ray classification

### Candidate 1: NIH ChestX-ray14 / CXR8

- Official source: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Access: Public Box folder.
- Reported usage requirement: provide a link to the original download site,
  acknowledge the NIH Clinical Center, and cite the CVPR 2017 paper.
- Approximate archive size: about 45 GB.
- Intended use: non-commercial educational ML classification prototype.
- Redistribution policy: do not commit the full dataset, patient-level metadata,
  or large derived artifacts to Git.
- Citation: to be recorded from the official README/paper before download.

### Candidate 2: CheXpert Small

- Official source: https://aimi.stanford.edu/datasets/chexpert-chest-x-rays
- Citation DOI: https://doi.org/10.71718/y7pj-4v93
- Access: Stanford AIMI / Redivis; requires review and acceptance of the
  applicable Stanford dataset agreement.
- Approximate Small archive size: about 11 GB.
- Intended use: non-commercial educational ML classification prototype.
- Known restriction reported in access materials: non-commercial research use,
  no redistribution/republication, no patient identification, citation required.
- Hosted-demo implication: do not publicly serve CheXpert images or derivatives.

## Workflow B — 3D segmentation

### Medical Segmentation Decathlon Task09 Spleen

- Official source: http://medicaldecathlon.com/
- Task: Task09_Spleen
- Reported contents: 61 CT volumes, 41 training and 20 testing.
- Reported license: CC BY-SA 4.0.
- Reported archive size: 1.6 GB.
- Intended use: non-commercial educational 3D segmentation pilot.
- Citation: Antonelli, M., et al. (2022). Medical Segmentation Decathlon.
  Nature Communications. DOI: 10.1038/s41467-022-30695-9.
- Before download: verify official access link, checksum if available,
  extracted size, and available local disk space.