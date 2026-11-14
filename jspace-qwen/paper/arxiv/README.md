# arXiv source

`main.tex` (+ `refs.bib`, `figures/`) is the arXiv-format write-up of this project. `main.pdf` is the
compiled paper and `main.bbl` the processed bibliography, which arXiv uses directly.

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

To submit, upload `main.tex`, `main.bbl` and `figures/`. Every number comes from a file in
`../../results/`. Results still being computed are marked `\pending{...}` and render in red;
they are filled in as the runs finish.
