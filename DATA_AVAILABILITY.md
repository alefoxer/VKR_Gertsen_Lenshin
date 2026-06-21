# Data Availability

The repository includes the code and demonstration artifacts required to reproduce the main experimental workflow. The compact baseline model and vocabulary files are included in the `models/` directory.

The generated experimental outputs are not stored in the repository by default because they can be reproduced by running:

```bash
python experiments/run_final_vkr_experiment.py
```

The extended manuscript experiment outputs are also not stored in the repository by default. They can be reproduced by running:

```bash
python experiments/run_manuscript_extended_experiment.py --steps 50 --repeats 5 --input-modes maximize_from_noise maximize_from_silence --seed-start 8000
```

For the manuscript, the generated directory under `outputs/manuscript_extended_experiment/<experiment_id>/` may be supplied as supplementary material or archived separately with the release.

Any additional local or private audio files used for testing are not publicly released if they may contain personal or non-public speech data.
