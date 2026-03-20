# SPMA

Support-Preserving Manifold Assimilation and follow-on manifold-continuation experiments live here.

Start with:

- [`main.py`](/home/henrykobs/SAE_CL/spma/main.py): single experiment run
- [`paper_suite.py`](/home/henrykobs/SAE_CL/spma/paper_suite.py): multiseed benchmark pipeline
- [`config.py`](/home/henrykobs/SAE_CL/spma/config.py): experiment config and method presets
- [`finetune.py`](/home/henrykobs/SAE_CL/spma/finetune.py): continual finetuning loop
- [`manifold_memory.py`](/home/henrykobs/SAE_CL/spma/manifold_memory.py): latent support / chart memory construction
- [`losses.py`](/home/henrykobs/SAE_CL/spma/losses.py): manifold and distillation losses

Canonical entrypoints:

- [`../scripts/run_spma.py`](/home/henrykobs/SAE_CL/scripts/run_spma.py)
- [`../scripts/run_spma_multiseed.py`](/home/henrykobs/SAE_CL/scripts/run_spma_multiseed.py)
- [`../scripts/run_spma_tune.py`](/home/henrykobs/SAE_CL/scripts/run_spma_tune.py)
