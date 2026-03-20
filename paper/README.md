# SPMA Paper Build Notes

The arXiv-style manuscript is in `main.tex`.

## Regenerating tables and figures

The paper assets are generated directly from the saved experiment outputs:

```bash
cd paper
python build_figures.py
```

This writes:

- figures in `paper/figures/`
- LaTeX tables in `paper/tables/`

The current asset builder expects:

- 5-seed novel-class runs under `../outputs/spma/paper_suite/`
- the 3-seed compatible-shift pilot under `../outputs/spma/tiny_imagenet_compatible_shift_plus3_1ep/`

## Compiling

```bash
cd paper
python build_figures.py
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Files

- `main.tex`: manuscript
- `references.bib`: bibliography
- `build_figures.py`: generates paper tables and figures from experiment outputs
- `spma_workshop_draft.md`: source prose draft used to create the LaTeX version
