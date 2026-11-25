# Research workspace walkthrough

The guest dashboard opens without sign-in. The team workspace requires an approved Google account. Both entry points are listed at the top of the [project README](../README.md).

| Workspace section | Example |
| --- | --- |
| Overview | A study card opens the corresponding cohort in the analysis workspace. |
| Datasets | Searching for `GSM2751834` identifies its development study. The CSV export follows the active filters. |
| Analysis | The spatial PCA view plots two principal components and recorded dose. Heatmaps, correlations and clustering show patterns within the chosen cohort. |
| Experiments | **Final evaluation** opens on whole blood + WBC gradient boosting. Its donor-balanced mean absolute error is **0.777 Gy** on 52 previously evaluated profiles from 12 donors. The model card and export include the uncertainty interval. |
| Predictions | Selecting a curated profile and model retrieves an estimate together with the recorded dose and training overlap. The team interface also supports compatible, reviewed uploads. |
| Assistant | Published answers link back to evidence. Guest questions and team tool-assisted conversations have separate access rules. |

The development comparison and bounded training jobs appear in separate tabs under Experiments. Their study holdouts answer a different question from the GSE90909 evaluation. New jobs leave the published final results intact.
