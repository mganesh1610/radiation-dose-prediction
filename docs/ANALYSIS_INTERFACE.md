# Analysis interface

The Analysis page uses the same catalogue, analysis requests, cached responses and scientific plotting functions as the other workspace views. `analysis-workspace.js` adds the visual layout, summary, study highlights, profile inspector and exports; `analysis-scene.mjs` is an optional Three.js presentation.

The spatial scene uses the existing PC1/PC2 scores on the ground plane and **recorded numerical radiation dose in Gy vertically**. Height is not PC3. It does not fit a PCA, transform an expression matrix, drop profiles, calculate a radiation prediction or alter scientific results. If a measured coordinate or dose is unavailable, or WebGL/modules cannot load, the original 2D chart remains available.

All original settings remain: four core cohorts, two expression representations and three gene-count choices. Heatmap scale and gene/profile ordering are unchanged. The Clustering tab retains its full saved six-study diagnostic controls. The Dose response & contrasts tab was removed at the user's request; archived statistical data and its original implementation are retained separately. The visible Analysis hero title/tagline were removed, and Methods/Export sit beside the five remaining tabs. Protected prediction/training/assistant routes and authentication remain separate.

The summary and inspector describe the currently displayed result. Changing a setting does not relabel the old result; the existing analysis button loads the matching result. Highlighting a study does not remove profiles or refit the analysis. Table search does not filter the plot or the CSV export. Snapshot JSON continues to use the original export handler, including heatmap display provenance. PNG export is available for the spatial canvas.

Three.js is pinned and served from the same origin. The optional renderer loads lazily and draws on demand; automatic rotation is opt-in, pauses off-page/when hidden, and respects reduced motion. The restrictive existing content security policy is retained.

Run the existing source and JavaScript checks, plus:

```sh
node --test tests/test_spatial_analysis.mjs
node tests/test_analysis_autoload.cjs
```

The tests check measured-coordinate preservation, recorded-dose height, missing-value fallback, stable control IDs, local module resolution and isolation of the existing request/export flow from presentation failures. Browser checks are additionally needed for rendering, responsive layout, pointer interaction and downloads. Passing software tests is not scientific or clinical validation.
