# Project Report

The formal PhoneWatch project report is written in LaTeX:

```bash
docs/phonewatch_project_report.tex
```

To compile it on a machine with TeX Live, MacTeX, or BasicTeX installed:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs docs/phonewatch_project_report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory docs docs/phonewatch_project_report.tex
```

Run the command twice so the table of contents and table list resolve correctly.
