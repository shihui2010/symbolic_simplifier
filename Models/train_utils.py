import torch
import os
import json
import shutil
from Data.halide_utils import HalideVocab


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")


def calibrate_hps(hps, vocab, decoder_name):
    if "complexity_split" not in hps:
        hps["complexity_split"] = hps["encoder"]["output_size"]
    else:
        assert hps["complexity_split"] <= hps["encoder"]["output_size"]

    hps["encoder"]["input_size"] = len(vocab)
    hps["decoder"]["input_size"] = len(vocab)
    hps["decoder"]["encoder_out_size"] = hps["complexity_split"]
    if hps["decoder_name"] in ["AttLSTM", "AttTreeDecoder"]:
        hps["decoder"]["attention_len"] = hps["input_len"]
    hps["decoder"]["encoder_hidden_size"] = hps["encoder"]["hidden_size"]
    if "Pointer" in decoder_name:
        hps["decoder"]["output_size"] = vocab.op_num + 11  # 11 basic tokens
        hps["decoder"]["oov_size"] = hps["encoder"]["input_size"] - hps["decoder"]["output_size"]
        if "Tree" not in decoder_name:
            hps["seq_len"] = hps["input_len"]
    else:
        hps["decoder"]["output_size"] = len(vocab)
    if "vocab_name" in hps:
        if hps["vocab_name"] == "Elementary":
            hps["encoder"]["N"] = 2
            hps["decoder"]["N"] = 2
        else:
            hps["encoder"]["N"] = 3
            hps["decoder"]["N"] = 3


def get_halide_vocab():
    vocab = HalideVocab()
    for token in ['x', 'y', 'z', 'w', 'u', 'v']:
        vocab.add_word(token)
    return vocab


class RLModelSaver(object):
    def __init__(self):
        self.performance = list()
        self.update = 0

    def save(self, model, logdir, depth, train_perf, test_perf, valid_perf, ruleset=None):
        self.update += 1
        save_path = os.path.join(logdir, f"D{depth}_{train_perf}_{test_perf}_{valid_perf}")
        removed = list()
        depth_matched = False
        for record in self.performance:
            d, train, test, valid, old_copy = record
            if d != depth:
                continue
            depth_matched = True
            if train >= train_perf and test >= test_perf and valid >= valid_perf:
                return
            elif train_perf >= train and test_perf >= test and valid_perf >= valid:
                if not (train_perf == train and test_perf == test and valid_perf == valid):
                    self.update = 0
                shutil.rmtree(old_copy)
                removed.append(record)
        if not depth_matched:
            self.update = 0
        for record in removed:
            self.performance.remove(record)

        self.performance.append((depth, train_perf, test_perf, valid_perf, save_path))
        model.save(save_path)
        if ruleset is not None:
            with open(save_path + "/rules.json", "w") as fp:
                json.dump(ruleset.dump(), fp, indent=2)
