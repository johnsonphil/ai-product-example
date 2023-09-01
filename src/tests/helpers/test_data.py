from pyspark.sql import SparkSession	
import warnings
warnings.filterwarnings(action='ignore', category=ResourceWarning)

from src.helpers.data import check_categories

spark = SparkSession.builder \
                    .appName('integrity-tests') \
                    .getOrCreate()


def test_check_categories():
    """
    Test the check_categories function.
    """
    # Create training and test data with some new categories
    train_data = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "category"])
    test_data = spark.createDataFrame([(4, "d"), (5, "e"), (6, "f")], ["id", "category"])

    # Test the function
    # new_rec = []
    # with pytest.warns(UserWarning) as record:
    #     check_categories(train_data, test_data)
    # for rec in record:
    #     print(rec)
    #     if rec.category == "UserWarning":
    #         new_rec.append(rec)
    # assert "Test data has new categories in column 'category' that are not present in the training data:" in str(new_rec[0].message)

    columns_w_new_cats = check_categories(train_data, test_data)
    assert len(columns_w_new_cats) == 2
