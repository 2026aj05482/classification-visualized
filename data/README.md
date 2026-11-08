# Dataset attribution

`breast_cancer_wisconsin.csv` contains the UCI Breast Cancer Wisconsin
(Diagnostic) dataset, loaded from the copy bundled with scikit-learn.

Original source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Citation: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5DW2B

Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0).
https://creativecommons.org/licenses/by/4.0/

Changes for this project: identifiers omitted; 30 measurements given descriptive
snake_case headers in original order; diagnosis encoded as 0 = benign and
1 = malignant (the inverse of scikit-learn's target encoding).
The prefixes mean, error and worst represent the mean, standard error and mean
of the three largest values respectively for each of ten nuclear measurements.
`test_data.csv` in the repository root is the untouched 20% stratified holdout.

The repository's MIT license covers project code; the original dataset retains
its CC BY 4.0 license.
