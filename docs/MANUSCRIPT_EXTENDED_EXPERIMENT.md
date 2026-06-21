# Manuscript Extended Experiment

## 1. Purpose

`experiments/run_manuscript_extended_experiment.py` supports the extended reproducibility experiment used for the GRADV manuscript. The experiment evaluates target-class audio pattern discovery across all available target words in the selected model vocabulary, multiple input initialization modes, and repeated random seeds.

The script is intended for manuscript-level analysis rather than for interactive demonstration. It runs a controlled set of optimization experiments, exports machine-readable summaries, produces article-ready figures, and reports segment-level analyses that can be inspected independently by reviewers.

## 2. Reproducibility Statement

The experiment is designed to make the manuscript results reproducible from the archived software release. All numerical values reported in the manuscript should be taken from the generated CSV or JSON outputs produced by the script. Values should not be copied from manual estimates, screen captures, or informal console observations.

The current archived release DOI is:

```text
10.5281/zenodo.20768314
```

This DOI should remain unchanged until a new GitHub/Zenodo release is created and Zenodo issues a new release DOI.

## 3. Vocabulary and Model Scope

The compact default baseline vocabulary contains eight Russian commands:

```text
да
нет
стоп
вперёд
назад
привет
включи
выключи
```

The vocabulary is configurable. However, additional Russian or English words require a model, adapter and vocabulary file that actually include these words. The script does not make an arbitrary word recognizable unless the selected model supports it.

The included compact baseline vocabulary should therefore be interpreted as a demonstration configuration for the available keyword-spotting model, not as a general Russian or English speech-recognition vocabulary.

## 4. Experiment Design

By default, the script loads the selected model and detects its vocabulary through the adapter. If no target words are provided on the command line, all words in the available vocabulary are evaluated.

The manuscript configuration evaluates:

- target words: all words in the selected model vocabulary;
- input modes: `maximize_from_noise` and `maximize_from_silence`;
- repeats: 5 repeated runs per target word and input mode;
- optimization steps: 50 steps in the manuscript command;
- seed handling: deterministic seeds starting from `--seed-start`, assigned by target, input mode and repeat index;
- optimization method: `gradient_ascent` by default;
- objective: `logit` by default;
- stopping criterion: the run records whether the configured `--goal-score` is reached; the default manuscript goal score is `0.85`;
- output directory structure: one experiment directory is created under `outputs/manuscript_extended_experiment/<experiment_id>/`.

The command-line interface also allows overriding `--targets`, `--steps`, `--repeats`, `--input-modes`, `--seed-start`, `--model`, `--method`, `--objective`, `--learning-rate`, `--max-delta`, `--l2-weight`, `--tv-weight`, `--goal-score`, `--output-dir` and `--quick`.

## 5. Quick Verification Command

```bash
python experiments/run_manuscript_extended_experiment.py --targets да нет --steps 8 --repeats 1 --quick
```

This command is intended only to verify that the environment, model loading path, optimization pipeline and export code work. It should not be used as the source of manuscript-level numerical claims.

## 6. Full Manuscript Experiment Command

```bash
python experiments/run_manuscript_extended_experiment.py --steps 50 --repeats 5 --input-modes maximize_from_noise maximize_from_silence --seed-start 8000
```

This command produces the outputs used for manuscript-level analysis. It evaluates all available vocabulary targets in the selected model, runs both input modes, performs repeated seeded optimization, exports per-run and aggregate tables, and creates the figure set used for manuscript preparation.

The expected output directory is:

```text
outputs/manuscript_extended_experiment/<experiment_id>/
```

Generated outputs are not committed to the repository by default. They should be archived separately, attached to a Zenodo release when appropriate, or provided as supplementary material for the manuscript.

## 7. Output Files

The script writes the following files in the experiment directory.

### `extended_experiment_summary.json`

Machine-readable summary of the full experiment. It includes the experiment configuration, vocabulary, target words, runtime information, aggregate run counts, paths to exported files, target-level summaries, input-mode summaries, segment-ablation results, weight-sensitivity results and model-comparison results when available. This file is the preferred source for reproducibility metadata.

### `extended_experiment_summary.csv`

Per-run table with one row per target word, input mode, repeat and seed. It includes original score, final score, score gain, success flags, goal-reaching status, runtime, exported segment paths and error messages if a run fails. This file can be used to audit individual runs and recompute aggregate statistics.

### `target_word_summary.csv`

Aggregate table grouped by target word. It reports the number of runs, completed and failed runs, success rate, mean original score, mean final score, standard deviation of final score, best final score, mean score gain, best seed, best input mode, exact and similar fragment counts, top-ranked segment interval, top-segment score summary and best-run output path. This file is suitable for manuscript tables comparing vocabulary targets.

### `input_mode_summary.csv`

Aggregate table grouped by input mode. It reports run count, success rate, mean final score, standard deviation of final score, mean score gain and best final score. This file supports manuscript comparisons between initialization strategies.

### `segment_ablation_summary.csv`

Segment-ranking ablation table for the best run of each target word. It compares the top temporal window selected by the integrated GRADV score and by individual component rankings. This file supports analysis of whether the selected temporal windows depend on one metric or on the combined scoring rule.

### `weight_sensitivity_summary.csv`

Weight-sensitivity table for the best run of each target word. It recomputes integrated segment rankings under alternative contribution-weight profiles without rerunning the model. This file supports analysis of the stability of top-ranked temporal windows under reasonable scoring-weight changes.

### `model_comparison_summary.csv`

Model-comparison table, generated when a comparison adapter is available for the selected targets. It summarizes selected-model and comparison-model behavior using the same exported metrics. If model comparison is unavailable for a particular configuration, the JSON report records that limitation rather than inventing values.

### `extended_experiment_report.md`

Article-oriented Markdown report containing the experiment configuration, hardware/runtime note, vocabulary and targets, target-word summary, input-mode summary, segment-ranking ablation, weight-sensitivity analysis, optional model-comparison summary, paths to figures and limitations. This report is useful for reviewer inspection and for drafting manuscript tables.

### `figures/target_word_final_scores.png`

Publication-resolution bar chart of mean final score by target word. It can be used to visualize which vocabulary targets produce higher optimized target-class scores under the selected configuration.

### `figures/target_word_score_gains.png`

Publication-resolution bar chart of mean score gain by target word. It can be used to compare how much optimization changes the target-class score for each command.

### `figures/input_mode_success_rates.png`

Publication-resolution bar chart of success rate by input mode. It can be used to compare `maximize_from_noise` and `maximize_from_silence` under the selected goal score.

### `figures/segment_ablation_top_windows.png`

Publication-resolution chart comparing the top temporal window selected by each segment-ranking method. It supports visual inspection of whether different ranking criteria select similar or different time intervals.

### `figures/weight_sensitivity_overlap.png`

Publication-resolution chart showing overlap between top-ranked windows under alternative weighting profiles and the current integrated top-three windows. It supports robustness analysis of the segment scoring weights.

### `figures/model_comparison.png`

Publication-resolution model-comparison chart, generated when model-comparison output is available. It can be used to compare mean final scores across the selected model and comparison adapter outputs.

## 8. Segment-Ranking Ablation

The segment-ranking ablation compares the top temporal windows selected by:

- integrated score;
- saliency-only ranking;
- occlusion-only ranking;
- isolated-score-only ranking;
- signal-change-only ranking.

The integrated score is the current GRADV segment-ranking formula. The ablation checks whether this ranking simply reproduces one individual metric or combines several criteria. For each target word, the analysis uses the exported segment metrics from the best run and records the top interval, top score and overlap with the integrated top-three intervals.

## 9. Weight-Sensitivity Analysis

The script recomputes segment rankings using alternative weighting profiles without rerunning the model. This means that the analysis is based on the exported segment metrics from the best run of each target word and does not create new optimization results.

The configured profiles are:

- `current`: saliency `0.35`, occlusion `0.25`, isolated `0.15`, signal change `0.25`;
- `saliency-heavy`: saliency `0.50`, occlusion `0.20`, isolated `0.10`, signal change `0.20`;
- `occlusion-heavy`: saliency `0.20`, occlusion `0.50`, isolated `0.10`, signal change `0.20`;
- `isolated-heavy`: saliency `0.20`, occlusion `0.20`, isolated `0.40`, signal change `0.20`;
- `balanced`: saliency `0.25`, occlusion `0.25`, isolated `0.25`, signal change `0.25`.

This analysis checks how stable the top-ranked temporal windows are under reasonable changes in contribution-score weights. Stable overlap suggests that the selected windows are not an artifact of a single narrow weighting choice.

## 10. Figures

Generated figures are intended for manuscript preparation and are saved at publication-friendly resolution. The script uses neutral styling, readable labels and no decorative color scheme. Figures are saved under:

```text
outputs/manuscript_extended_experiment/<experiment_id>/figures/
```

The figures should be interpreted together with the CSV/JSON files, which provide the underlying numerical values.

## 11. Limitations

- Results are model-dependent.
- Results are vocabulary-dependent.
- Results depend on optimization settings and random seeds.
- The compact vocabulary is a demonstration configuration.
- Arbitrary Russian or English words require a corresponding trained or configured model, adapter and vocabulary file.
- The script does not validate industrial ASR systems.
- The script does not perform multi-channel microphone-array processing.

## 12. How to Cite / Manuscript Use

The software release DOI should be cited in the manuscript. Keep the current archived release DOI unchanged until a new Zenodo release DOI is issued:

```text
10.5281/zenodo.20768314
```

When reporting experiment results, cite the release and identify the exact generated experiment directory or supplementary archive used to produce the manuscript tables and figures.

## 13. Reviewer Checklist

- Run `python scripts/smoke_test.py`.
- Run `python scripts/verify_core.py`.
- Run the quick extended experiment command.
- Inspect `extended_experiment_summary.json` and the generated CSV files.
- Compare `extended_experiment_report.md` with the manuscript tables.
- Check generated figures under `figures/`.
