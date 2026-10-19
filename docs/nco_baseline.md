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

| Setting                               | Value                                                                     |
|---------------------------------------|---------------------------------------------------------------------------|
| Core                                  | Largest connected component of the 10-core                                |
| Nodes / edges                         | 2,896 / 58,466                                                            |
| Positive / negative / zero-sign edges | 52,807 / 4,190 / 1,469                                                    |
| Initial negative fraction f           | 0.1 to 0.9, step 0.1, plus 0.42, 0.44, 0.46, 0.48, 0.52, 0.54, 0.56, 0.58 |
| Realizations                          | 20 per model and fraction; 680 runs total                                 |
| Initialization                        | Random assignment with matching initial states across models              |
| Negative node count                   | Nearest integer to N times f; halves rounded up                           |
| Seed / update limit                   | 0 / 1,000                                                                 |

## Results

All 680 runs reach steady states. No cycles or update-limit stops. The table below
shows the original nine-point sweep, whose 360 outcomes are preserved unchanged.
`s_1` is the final negative node fraction.

| Initial fraction f | Unsigned mean s_1 | Signed mean s_1 |
|--------------------|-------------------|-----------------|
| 0.1                | 0.0000            | 0.0038          |
| 0.2                | 0.0000            | 0.0038          |
| 0.3                | 0.0013            | 0.0054          |
| 0.4                | 0.0060            | 0.0109          |
| 0.5                | 0.5484            | 0.5945          |
| 0.6                | 0.9970            | 0.9928          |
| 0.7                | 0.9993            | 0.9951          |
| 0.8                | 0.9998            | 0.9957          |
| 0.9                | 1.0000            | 0.9963          |

At f=0.5, sample standard deviations are 0.4746 unsigned and 0.4646 signed.
Model differences are modest. This sweep does not establish a critical-point shift.
Opinions are simulated; the core is not a random sample of Reddit.

## Component measurements and refinement

The refined sweep appends eight fractions to the original nine, keeping 20 paired
initial states per fraction. Seed assignment still follows the recorded fraction
order and realisation index. Appending points preserves all 360 original outcomes;
sorting or inserting fractions before generating seeds would change them.

For each opinion we save the largest and second-largest connected component,
divided by the full original core size N. `g_minus` and `g_plus` retain their
existing meanings. `g_minus_second` and `g_plus_second` refer to the second-largest
component of the same opinion, rather than the other opinion. Missing components
give zero, and zero-sign edges remain part of the fixed unsigned reference graph.

The component figures show each realisation and its group mean. Separate rows
show the two component ranks and the number of opinion nodes outside the largest
component. The latter is the opinion's total node count minus its largest
component size. Each row has its own y scale to keep small fragments visible.
Horizontal jitter is for visibility only; the CSV stores the exact f values.

At f=0.5, the original unsigned runs split into nine outcomes below 10% negative
and eleven above 90% negative. Their mean does not represent a typical balanced
steady state. Individual points make this distinction visible.

The finer grid resolves the transition more clearly. The counts below use a
negative node fraction greater than one half as a negative-majority outcome:

| Initial fraction f | Unsigned negative-majority runs / 20 | Signed negative-majority runs / 20 |
|--------------------|--------------------------------------|------------------------------------|
| 0.46               | 0                                    | 0                                  |
| 0.48               | 2                                    | 1                                  |
| 0.50               | 11                                   | 12                                 |
| 0.52               | 19                                   | 18                                 |
| 0.54               | 20                                   | 20                                 |

These small samples describe the observed transition and do not establish a
signed-versus-unsigned critical-point shift.

The pre-refinement artifacts for this checkout are retained under
`results/nco/baseline_coarse/`. The simulator, core, seed, realisation count, and
update limit stay fixed for the refinement.

## Outputs

Files under `results/nco/`:

| File                                                | Contents                                                                                                                                  |
|-----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `fraction_sweep.csv`                                | Raw opinion fractions, both component ranks per opinion, replay seeds, update counts, stop reasons, and cycle lengths.                    |
| `fraction_summary.csv`                              | Means/sample standard deviations for raw fractions and both component ranks, and proportions of runs by stop reason.                      |
| `phase_transition.png`, `.pdf`                      | Model comparison with sample standard deviation bars.                                                                                     |
| `opinion_gcc_phase_transition.png`, `.pdf`          | Negative opinion: individual runs and means for the largest component, second-largest component, and nodes outside the largest component. |
| `positive_opinion_gcc_phase_transition.png`, `.pdf` | The same measurements for the positive opinion.                                                                                           |                 |
| `metadata.json`                                     | Input checksum, parameters, conventions, and package versions.                                                                            |
| `nodes.json`                                        | Matrix node order.                                                                                                                        |

Summaries group steady states, cycles, and update-limit runs separately. Separate
plot columns show terminal phases/states when present. Error bars in the
raw-fraction figure represent run variability; the component figures show the
individual outcomes directly.