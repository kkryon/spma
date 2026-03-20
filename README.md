# spma

## Top-Level Layout

```text
SAE_CL/
  scripts/                    Canonical runnable entrypoints
  docs/                       Repo-level navigation notes
  spma/                       Shared-manifold continual learning
  rg_gpft/                    Region-gated geometry-preserving fine-tuning
  continual_sae_experiment/   SAE regularization experiments
  ig_vae_replay/              VAE replay experiments
  paper/                      LaTeX and figure-generation assets
  data/                       Downloaded datasets
  outputs/                    Experiment outputs
  checkpoints/                Saved checkpoints
```

## Navigation

If you are starting fresh:

1. Read [`docs/REPO_MAP.md`](/home/henrykobs/SAE_CL/docs/REPO_MAP.md).
2. Pick the project package you care about.
3. Use the matching script in [`scripts/`](/home/henrykobs/SAE_CL/scripts) or the root compatibility wrapper.

## Notes

- [`sitecustomize.py`](/home/henrykobs/SAE_CL/sitecustomize.py) and [`scripts/_bootstrap.py`](/home/henrykobs/SAE_CL/scripts/_bootstrap.py) help local execution find the repo and optional `.venv`.
- Large artifacts are intentionally kept in [`outputs/`](/home/henrykobs/SAE_CL/outputs) and [`checkpoints/`](/home/henrykobs/SAE_CL/checkpoints), separate from the source packages.
