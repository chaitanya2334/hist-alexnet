import os
import time
import operator
import numpy as np
import sys

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix


class Evaluator(object):
    def __init__(self, name, label2id, pure_labels=None):
        self.name = name
        self.label2id = label2id

        self.cost_sum = 0.0
        self.avg_fscore = 0.0
        self.acc = 0.0
        self.true = []
        self.pred = []
        self.start_time = time.time()
        self.total_samples = 0
        self.conf = None
        self.report = None
        self.p = 0.0
        self.r = 0.0
        self.f = 0.0

        if pure_labels is None:
            self.pure_labels = self.label2id
        else:
            self.pure_labels = pure_labels

    def append_data(self, cost, pred_label_ids, true_label_ids):
        if isinstance(pred_label_ids, list):
            self.total_samples += len(pred_label_ids)
            self.pred.extend(pred_label_ids)
            self.true.extend(true_label_ids)
        else:
            self.total_samples += 1
            self.pred.append(pred_label_ids)
            self.true.append(true_label_ids)

        self.cost_sum += cost

    def remove_nones(self):
        none_idx = [i for i, l in enumerate(self.true) if l is None]
        for idx in sorted(none_idx, reverse=True):
            del self.pred[idx]
            del self.true[idx]

        assert len(self.pred) == len(self.true)

        assert None not in self.pred
        assert None not in self.true

    def gen_results(self):

        assert len(self.true) == len(self.pred) != 0, "{0}, {1}".format(len(self.true), len(self.pred))
        target = sorted(self.pure_labels, key=lambda k: self.pure_labels[k])
        print(target)
        self.remove_nones()
        self.report = classification_report(self.true, self.pred, labels=range(len(target)), target_names=target,
                                            digits=6)
        self.p, self.r, self.f, self.s = precision_recall_fscore_support(self.true, self.pred, average="macro")
        self.acc = accuracy_score(self.true, self.pred)
        self.conf = confusion_matrix(self.true, self.pred)
        return self.report

    def print_results(self):
        print("{0}_total_samples: {1}".format(self.name, self.total_samples))
        print("{0}_cost_sum: {1}".format(self.name, self.cost_sum))
        print("{0}_acc: {1}".format(self.name, self.acc))
        print("{0}_Classification Report".format(self.name))
        print(self.report)
        print("{0}_confusion matrix".format(self.name))
        print(self.conf)
        print("{0}_precision: {1}".format(self.name, self.p))
        print("{0}_recall: {1}".format(self.name, self.r))
        print("{0}_fscore: {1}".format(self.name, self.f))

    def write_results(self, filename, text, spec='a'):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, spec, encoding='utf-8') as f:
            f.write("-" * 40 + "\n")
            f.write(text + "\n")
            f.write("-" * 40 + "\n")
            f.write("{0}_total_samples: {1}\n".format(self.name, self.total_samples))
            f.write("{0}_cost_sum: {1}\n".format(self.name, self.cost_sum))
            f.write("{0}_acc: {1}".format(self.name, self.acc))
            f.write("{0}_Classification Report\n".format(self.name))
            f.write(self.report)
            f.write("{0}_confusion_matrix\n".format(self.name))
            f.write(np.array_str(self.conf))
            f.write("{0}_precision: {1}\n".format(self.name, self.p))
            f.write("{0}_recall: {1}\n".format(self.name, self.r))
            f.write("{0}_fscore: {1}\n".format(self.name, self.f))

    def verify_results(self):
        if np.isnan(self.cost_sum) or np.isinf(self.cost_sum):
            sys.stderr.write("ERROR: Cost is NaN or Inf. Exiting.\n")
            exit()
