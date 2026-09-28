# DS Engineer test task

Solutions for the three parts of the test task. Each folder has its own README with setup
and usage.

| Task | Folder | Summary |
|---|---|---|
| 1. Counting islands | [task1_islands](task1_islands) | Iterative BFS flood fill with 4-connectivity, O(M·N) time |
| 2. Regression on tabular data | [task2_regression](task2_regression) | EDA shows `target = feature_6² + feature_7`; linear regression on these two features, validation RMSE ≈ 0 |
| 3. MNIST classifier (OOP) | [task3_mnist](task3_mnist) | `DigitClassifier` with a common interface for CNN, Random Forest and random models |

Predictions for `hidden_test.csv` are in
[task2_regression/predictions.csv](task2_regression/predictions.csv).

All code was tested with Python 3.13.
