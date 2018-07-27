from unittest import TestCase

from dataset.bic_dataset import BIC_Dataset


class TestBIC_Dataset(TestCase):
    def test__label_by_id(self):
        bic_dataset = BIC_Dataset("../../hist-infogan/bic_tiled_data/Training_data",
                                  "../../hist-infogan/bic_tiled_data/Test_data",
                                  ["Benign", "In Situ", "Invasive", "Normal"])

        print(bic_dataset._label_by_id(bic_dataset.train_lbls, str(210)))

    def test_test_data(self):
        bic_dataset = BIC_Dataset("../../hist-infogan/bic_tiled_data/Training_data",
                                  "../../hist-infogan/bic_tiled_data/Test_data",
                                  {"Benign": 0, "In situ": 1, "Invasive": 2, "Normal": 3})

        for i, l, id, tid in bic_dataset.test:
            print(id)
