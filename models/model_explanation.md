# Model Explanation

- Selected model: `decision_tree`
- Holdout accuracy: `1.0000`
- Holdout precision: `1.0000`
- Holdout recall: `1.0000`
- Holdout F1-score: `1.0000`
- Cross-validation accuracy: `1.0000 +/- 0.0000`

## Model Comparison
```text
         model_name  cv_accuracy_mean  cv_accuracy_std  holdout_accuracy  holdout_precision  holdout_recall  holdout_f1_score                                                                         best_params
      decision_tree             1.000         0.000000             1.000               1.00        1.000000          1.000000 {'model__class_weight': None, 'model__max_depth': 4, 'model__min_samples_leaf': 20}
logistic_regression             0.869         0.005612             0.865               0.75        0.666667          0.705882                                      {'model__C': 0.5, 'model__class_weight': None}
```

## Top Features By Permutation Importance
- `attendance_rate` changed accuracy by about `0.2056` when shuffled.
- `gpa` changed accuracy by about `0.1590` when shuffled.
- `engagement_score` changed accuracy by about `0.1362` when shuffled.
- `stress_level` changed accuracy by about `0.0915` when shuffled.

## Top Features By Model-Specific Importance
- `gpa` carries a relative importance score of `0.2608`.
- `engagement_score` carries a relative importance score of `0.2582`.
- `attendance_rate` carries a relative importance score of `0.2413`.
- `stress_level` carries a relative importance score of `0.2397`.

## Interpretation
- Attendance, GPA, engagement, and stress level are the strongest global signals in the selected model.
- The engineered features help compress multiple academic and behavioral signals into more stable summary indicators.
- A perfect or near-perfect score on this dataset suggests the labels may be highly rule-driven or synthetic, so real-world performance could be lower on noisier data.