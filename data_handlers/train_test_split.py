from sklearn.model_selection import StratifiedKFold


def stratified_k_fold_split(
    train_index, y, n_splits=5, shuffle=True, random_state=42
):
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    for train_index, test_index in skf.split(train_index, y):
        yield train_index, test_index
