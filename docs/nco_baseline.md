# NCO baseline

## Rules

- States: -1 or +1. All nodes update synchronously, with one self vote.
- Positive links favor agreement; negative links favor disagreement.
- Opinion ties preserve the current state. Isolated nodes never change.
- Equal positive/negative interaction counts give zero signed influence.
- Simulation stops at a steady state, a repeated cycle, or the update limit.

## Run

After setup and characterisation in the [README](../README.md):

```bash
uv run python -m mdacn nco --realizations 20 --seed 0 --max-steps 1000
```

| Setting | Value |
| --- | --- |
| Core | Largest connected component of the 10-core |
| Nodes / edges | 2,896 / 58,466 |
| Positive / negative / zero-sign edges | 52,807 / 4,190 / 1,469 |
| Initial negative fraction f | 0.1 to 0.9, step 0.1 |
| Realizations | 20 per model and fraction; 360 runs total |
| Initialization | Random assignment with matching initial states across models |
| Negative node count | Nearest integer to N times f; halves rounded up |
| Seed / update limit | 0 / 1,000 |

## Results

All 360 runs reach steady states. No cycles or update-limit stops.
`s_1` is the final negative node fraction.

| Initial fraction f | Unsigned mean s_1 | Signed mean s_1 |
| --- | --- | --- |
| 0.1 | 0.0000 | 0.0038 |
| 0.2 | 0.0000 | 0.0038 |
| 0.3 | 0.0013 | 0.0054 |
| 0.4 | 0.0060 | 0.0109 |
| 0.5 | 0.5484 | 0.5945 |
| 0.6 | 0.9970 | 0.9928 |
| 0.7 | 0.9993 | 0.9951 |
| 0.8 | 0.9998 | 0.9957 |
| 0.9 | 1.0000 | 0.9963 |

At f=0.5, sample standard deviations are 0.4746 unsigned and 0.4646 signed.
Model differences are modest. This sweep does not establish a critical-point shift.
Opinions are simulated; the core is not a random sample of Reddit.

## Outputs

Files under `results/nco/`:

| File | Contents |
| --- | --- |
| `fraction_sweep.csv` | Individual outcomes, replay seeds, update counts, stop reasons, and cycle lengths. |
| `fraction_summary.csv` | Means, sample standard deviations, and proportions of runs by stop reason. |
| `phase_transition.png`, `.pdf` | Model comparison with sample standard deviation bars. |
| `metadata.json` | Input checksum, parameters, conventions, and package versions. |
| `nodes.json` | Matrix node order. |

Steady-state summaries exclude cycles and update-limit runs. Separate plot panels
show their terminal states when present. Error bars represent run variability.
