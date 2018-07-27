import gc
import random
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler
from tqdm import tqdm


class FastWeightedRandomSampler(Sampler):
    r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        weights (list)   : a list of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, class_weights, sample_labels, max_nsamples, replacement=True, get_all=True):
        # simple data sanity checks
        if not isinstance(max_nsamples, int) or isinstance(max_nsamples, bool) or \
                max_nsamples <= 0:
            raise ValueError("num_samples should be a positive integeral "
                             "value, but got max_nsamples={}".format(max_nsamples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))

        # binned_samples[label] = list of sample indices
        self.binned_samples = self.bin(sample_labels)
        self.weights = [class_weights[k] for k in sorted(class_weights.keys())]
        self.max_nsamples = max_nsamples
        self.replacement = replacement
        self.samples_left = max_nsamples
        self.get_all = get_all
        self.num_samples = max_nsamples

    @staticmethod
    def bin(sample_labels):
        # we want to create a dict where the keys are the unique classes for every sample,
        # and the value for each class label is a list of all samples that share that label.
        d = defaultdict(list)
        for i, w in enumerate(tqdm(sample_labels)):
            d[w].append(i)
        return d

    def __iter__(self):
        self.multi = torch.multinomial(torch.DoubleTensor(self.weights), self.max_nsamples, True)
        self.samples_left = self.max_nsamples
        self.num_samples = self.max_nsamples
        self.weights_left = self.weights
        self.binned_sampler = [random.sample(v, len(v)) for k, v in self.binned_samples.items()]
        if not self.get_all:
            self.num_samples = self.get_count()
            self.samples_left = self.num_samples
        return self

    def get_count(self):
        c = defaultdict(int)
        ret = 0
        for w in self.multi.tolist():
            c[w] += 1
            ret += 1
            if c[w] >= len(self.binned_sampler[w]):
                break
        return ret

    def __next__(self):
        if self.samples_left > 0:
            w = self.multi[self.num_samples - self.samples_left].item()
            # if all the samples having weight = w are empty,
            # remove the weight from multinomial and try again
            
            while self.get_all and not self.binned_sampler[w]:
                # update multi
                print("{0} is empty".format(w))
                self.weights_left[w] = 0
                self.multi = torch.multinomial(self.weights_left, self.num_samples, True)
                w = self.multi[self.num_samples - self.samples_left].item()

            self.samples_left -= 1
            try:
                return self.binned_sampler[w].pop()
            except IndexError:
                print(w, self.samples_left)

        else:
            raise StopIteration

    def __len__(self):
        return self.num_samples
