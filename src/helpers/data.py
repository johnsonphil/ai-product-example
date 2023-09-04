import warnings


def check_categories(train_data, test_data):
    """
    Check if the categories in the test data are present in the categories of the training data.
    """
    columns = []

    for col in test_data.columns:
        train_categories = set(train_data.select(col).distinct().rdd.flatMap(lambda x: x).collect())
        test_categories = set(test_data.select(col).distinct().rdd.flatMap(lambda x: x).collect())
        new_categories = test_categories - train_categories
        if new_categories:
            warnings.warn(UserWarning("Test data has new categories in column '{}' that are not present in the training data: {}".format(col, new_categories)))
            columns.append(col)
    return columns
