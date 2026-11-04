# Advanced Statistics v11 audit

Implemented a design-aware statistics layer.

## Added
- Experimental-design audit for Animal/Subject, Group, Timepoint/Session and Dam/Cage.
- Subject-level aggregation for repeated observations.
- Two-timepoint paired t-test / Wilcoxon signed-rank analysis.
- Multi-timepoint complete-case Friedman analysis.
- Subject-random-intercept mixed-effects models using statsmodels when available.
- Optional Dam/Cage variance component in mixed-effects models.
- Between-group analysis reuses the existing Welch/Mann-Whitney/ANOVA/Kruskal-Wallis engine with BH-FDR correction.
- Effect estimate and 95% CI for paired changes and Cohen's d bootstrap CI utility.
- Advanced Statistics UI with Design Audit, Subject Level, Repeated Measures, Mixed Effects and Between Groups tabs.
- `Advanced_Statistics.xlsx` plus CSV outputs.

## Scientific safeguards
- No missing Group, Subject, Timepoint or Dam/Cage relationships are invented.
- Repeated observations are not treated as independent animals when subject IDs exist.
- Dam/Cage is not silently treated as an independent experimental unit.
- Mixed-model failures are reported explicitly instead of falling back silently.
- Final model selection still requires researcher/statistician validation for the actual experimental design.
