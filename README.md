# Aircraft Recoverability Expert System

This repository contains the public reproducibility code for the manuscript:

> Evidence-Governed Human--AI Collaborative Network for Aircraft Turnaround Recovery

The study asks how a governed human-AI network should allocate review, escalation, deferral, and abstention rights when disrupted aircraft chains compete for a limited specialist queue. The target is the smallest feasible queue that captures supported recovery opportunities while keeping unreviewed opportunity risk below a declared level. The code reconstructs aircraft turnaround chains, builds compatible historical continuation evidence, learns joint focal-failure and feasible-continuation opportunity, and certifies review-allocation and bounded-deferral policies under finite-sample risk control. Bounded deferral removes a case from specialist recovery review; flight execution decisions remain with established operating roles.

## Data acquisition

The analysis uses United States Bureau of Transportation Statistics Airline On-Time Performance records as the observational source for evaluating the evidence-capacity-authority policy. Source flight records are excluded from this repository. Download the monthly files directly from:

- https://transtats.bts.gov/ONTIME/
- https://transtats.bts.gov/PREZIP/

Place source files under `data/`. Generated results are written under `results/`. Both directories are excluded from version control.

The reported review capacities are normalized fractions of eligible cases. A deployment unit can translate a risk-controlled fraction into service minutes by supplying the expected review time for each case and the time available in the decision window. Reviewer acceptance, recommendation challenge, service duration, and trust are downstream interaction measurements; they are recorded by a later approved human-in-the-loop deployment study.

## Environment

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run modules from `src/recoverability` so that the local module imports resolve consistently.

## Reproducibility modules

The public snapshot contains the modules used for the current manuscript:

- `operational_records.py`: record ingestion, turnaround reconstruction, focal outcomes, and observed-path scoring;
- `turnaround_reconstruction.py`: multi-period turnaround reconstruction;
- `observed_path_calibration.py` and `calibration_assessment.py`: chronological calibration and probability diagnostics;
- `relational_features.py`, `temporal_relation_model.py`, and `multitask_recoverability.py`: typed temporal relations and the structure-constrained multi-task opportunity model;
- `compatible_history.py` and `prior_record_recoverability.py`: endpoint-closed historical continuation evidence;
- `completed_record_certificate.py` and `prior_record_certificate.py`: finite-sample risk control for completed-data and prior-data decisions;
- `support_conditioned_robust_selection.py`: exact rectangular worst-case envelopes, scalar shift envelopes, and minimax grid-capacity selection;
- `public_record_robustness_audit.py`: airport- and carrier-cluster bootstrap audit of supported-opportunity capture;
- `frontier_methods/` and `frontier_comparison.py`: common-task adaptations of the recent comparison families reported in the manuscript;
- `submission_displays.py`: tables and figures generated only from saved result files.

## Evaluation order

The analysis follows a chronological data boundary. Training, tuning, risk calibration, and validation segments are separated before model selection and policy evaluation. A typical reproduction proceeds through the following stages:

1. reconstruct annual aircraft turnaround chains from the public monthly files;
2. estimate and calibrate observed-path recovery probabilities;
3. construct compatible continuation evidence and multi-task temporal relation scores;
4. certify completed-record review capacity;
5. construct prior-record scores using only endpoint-closed earlier histories;
6. certify the prior-record review and low-risk deferral boundary;
7. evaluate the recent comparison families under the same evidence and outcome definitions;
8. generate manuscript displays from saved results.

Use each module's `--help` option for file and policy arguments. The manuscript and Supplementary Material define the fixed annual splits, compatibility rules, risk targets, capacity grid, and evaluation metrics.

## Reproducibility boundary

The repository provides executable analysis code and excludes source data, derived episode-level outputs, local configurations, and manuscript-development history. Public flight records must be obtained from the official source. The code does not contain private airline operational data.

## License

The code is released under the MIT License.
