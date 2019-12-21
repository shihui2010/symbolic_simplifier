import os
import re
import glob
import json
import numpy as np
from Data.halide_utils import HalideVocab
from Data.data_utils import tokenize, parse_expression


class RLBatcher(object):
    def __init__(self, file_name, split=0.2, n_sample=None):
        self.vocab, self._data = self.load(file_name, n_sample=n_sample)
        data_size = len(self._data)
        print(file_name, data_size)
        idx_pool = np.arange(data_size)
        np.random.seed(2019)
        np.random.shuffle(idx_pool)

        self.test_size = int(data_size * split)
        self.train_size = data_size - 2 * self.test_size
        self.train_pool = idx_pool[:self.train_size]
        self.test_pool = idx_pool[self.train_size: self.train_size + self.test_size]
        self.valid_pool = idx_pool[-self.test_size:]
        self.baseline_reward = 0
        self._reward_update = 0
        self._reward_n = 0

    @staticmethod
    def load(fname, n_sample=None):
        vocab = HalideVocab()
        for i in "0123":
            vocab.add_word(f"c{i}")
            vocab.add_word(f"v{i}")
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RL_Data", fname)) as fp:
            data = list()
            for lid, line in enumerate(fp):
                lhs, rhs = json.loads(line)
                lhs = parse_expression(lhs, vocab)
                rhs = parse_expression(rhs, vocab)
                data.append((lhs, rhs))
        if n_sample is not None and n_sample < len(data):
            np.random.seed(2019)
            idx = np.random.choice(np.arange(len(data)), n_sample, replace=False)
            return vocab, [data[i] for i in idx]
        return vocab, data

    def update_reward(self, reward):
        self._reward_update *= self._reward_n
        self._reward_update += reward
        self._reward_n += 1
        self._reward_update /= self._reward_n

    def finalize_epoch(self):
        self.baseline_reward = self._reward_update
        self._reward_update = 0
        self._reward_n = 0

    def sos(self, batch_size):
        return np.ones([1, batch_size]) * self.vocab.start()

    def eos(self, batch_size):
        return np.ones([1, batch_size]) * self.vocab.end()

    def _iter_pools(self, pool):
        np.random.shuffle(pool)
        for i in pool:
            yield self._data[i]

    def train_samples(self):
        yield from self._iter_pools(self.train_pool)
        np.random.shuffle(self.train_pool)

    def test_samples(self):
        yield from self._iter_pools(self.test_pool)
        np.random.shuffle(self.test_pool)

    def valid_samples(self):
        yield from self._iter_pools(self.valid_pool)
        np.random.shuffle(self.valid_pool)

    @staticmethod
    def compare(ws, tree):
        if type(tree) != list:
            tree = tree.to_tokens()
        if type(ws) != list:
            ws = ws.to_tokens()
        if len(ws) != len(tree):
            return False
        for w1, w2 in zip(ws, tree):
            if w1 != w2:
                return False
        return True


class CurriculumBatcher(object):
    def __init__(self, vocab_name, n_var, split=0.2, n_sample=None):
        self.data = dict()
        self.vocab = None
        for fname in glob.glob(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "RL_Data", vocab_name + "_var" + str(n_var) + "*.json")):
            file_in = fname.split("/")[-1]
            sub_batcher = RLBatcher(file_in.split("/")[-1], split=split, n_sample=n_sample)
            if self.vocab is None:
                self.vocab = sub_batcher.vocab
            depth = int(re.findall("d([2-6])", file_in)[-1])
            if file_in.endswith("in.json"):
                mode = "in"
            elif file_in.endswith("out.json"):
                mode = "out"
            else:
                mode = "none"
            self.data.setdefault(depth, dict())[mode] = sub_batcher

    def level(self, depth, train_samples=True, test_samples=True, val_samples=True):
        for d in self.data:
            if d > depth:
                continue
            for mode in self.data[d]:
                if train_samples:
                    yield from self.data[d][mode].train_samples()
                if test_samples:
                    yield from self.data[d][mode].test_samples()
                if val_samples:
                    yield from self.data[d][mode].valid_samples()

    def sos(self, batch_size):
        return np.ones([1, batch_size]) * self.vocab.start()

    def eos(self, batch_size):
        return np.ones([1, batch_size]) * self.vocab.end()


class PipelineBatcher(object):
    def __init__(self):
        self.vocab = HalideVocab()
        self.train_set = self.__load("train")
        self.test_set = self.__load("test")
        self.val_set = self.__load("val")
        self.train_pool = np.arange(len(self.train_set))
        print("Vocabulary size is", len(self.vocab))

    def __load(self, split):
        fname = os.path.join(os.path.dirname(__file__), "simplify", split + ".json")
        exprs = list()
        print(f"reading file: {fname}")
        with open(fname) as fp:
            data = json.load(fp)
        lengths = 0
        sizes = 0
        for i, _ in data:
            in_seq = tokenize(i, self.vocab)
            if "%" in in_seq:
                continue
            LUT = dict()
            for tid, token in enumerate(in_seq):
                const_value = self.vocab.to_const(token)
                if const_value is not None:
                    val = "c" + str(len(LUT))
                    for v in LUT:
                        if LUT[v] == const_value:
                            val = v
                    LUT[val] = const_value
                    in_seq[tid] = val
                    token = val
                self.vocab.add_word(token)

            tree_in = parse_expression(in_seq, self.vocab)
            exprs.append([tree_in, LUT])
            lengths += len(in_seq)
            sizes += tree_in.size()

        print(f"{len(exprs)} pairs of expression loaded")
        print(f"Avg length: {lengths /len(exprs)} ")
        print(f"Avg size: {sizes / len(exprs)}")
        return exprs

    def next_train(self):
        for i in self.train_pool:
            yield self.train_set[i]
        np.random.shuffle(self.train_pool)

    def next_test(self):
        for i in self.test_set:
            yield i

    def next_val(self):
        for i in self.val_set:
            yield i


if __name__ == "__main__":
    batcher = PipelineBatcher()
    idx = 0
    for i in batcher.next_test():
        print(i)
        idx += 1
        if idx == 10:
            break
    print(len(batcher.vocab))
    for i in batcher.vocab.word2id:
        print(i)
