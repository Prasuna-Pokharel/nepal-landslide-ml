
import unittest

from src.data_processing import (
    get_data_quality_report,
    load_data,
)


class TestDataProcessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_data()

    def test_dataset_is_not_empty(self):
        self.assertGreater(len(self.df), 0)
        self.assertGreater(len(self.df.columns), 0)

    def test_expected_dataset_dimensions(self):
        self.assertEqual(len(self.df), 12838)
        self.assertEqual(len(self.df.columns), 17)

    def test_processed_data_has_no_missing_values(self):
        self.assertEqual(int(self.df.isna().sum().sum()), 0)

    def test_processed_data_has_no_duplicate_rows(self):
        self.assertEqual(int(self.df.duplicated().sum()), 0)

    def test_feature_counts(self):
        numeric_count = len(
            self.df.select_dtypes(include="number").columns
        )
        categorical_count = len(
            self.df.select_dtypes(exclude="number").columns
        )

        self.assertEqual(numeric_count, 14)
        self.assertEqual(categorical_count, 3)

    def test_quality_report_matches_dataset(self):
        report = get_data_quality_report(self.df)

        self.assertEqual(report["rows"], len(self.df))
        self.assertEqual(report["columns"], len(self.df.columns))
        self.assertEqual(report["missing_values"], 0)
        self.assertEqual(report["duplicate_rows"], 0)
        self.assertEqual(report["numeric_features"], 14)
        self.assertEqual(report["categorical_features"], 3)


if __name__ == "__main__":
    unittest.main()
