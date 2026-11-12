
## Scope

The main implementation is the `analysis_v126` package. It contains:

- TabM-based prediction and sparse-support response adaptation;
- historical training, outer-year prediction, and evaluation entry points;
- the Gaussian design and selection utilities in `src/g2f_design`;
- data preparation and outer-worker helpers in `operations`;
- fixed experiment specifications and the projection matrix in `design_v126`.

The data contract is enforced by hashes, year checks, role checks, label isolation checks, and support-set validation. Data files must be supplied separately under the paths passed to the command-line entry points.

## Requirements

Python 3.10 or newer is recommended. Install the required dependency set with:

```bash
python -m pip install -r requirements.txt
```

The package metadata is also provided in `pyproject.toml`.

## Tests

Run the complete local validation suite with:

```bash
python -m pytest -q tests
```

The suite covers the response update, training, data contracts, outer prediction and evaluation, Gaussian design utilities, and selection geometry.

## Reproduction entry points

Use the command-line help before running a stage:

```bash
python -m analysis_v126.train --help
python -m analysis_v126.outer_predict --help
python -m analysis_v126.outer_evaluate --help
python -m analysis_v126.core_summary --help
python operations/prepare_v126_data.py --help
```

The training and evaluation scripts require the corresponding prepared data-contract files, role files, support labels, and checkpoints. These inputs are intentionally kept outside this repository. The fixed projection and experiment tables required by the code are included in `design_v126`.

## Review release

This snapshot is frozen as `v1.0-review`. The code is supplied for method inspection and reproducibility assessment; numerical results should be regenerated from the permitted data sources and the recorded experiment contract.
