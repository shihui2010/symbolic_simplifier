import os
import json
import torch
import torch.optim as optim
import numpy as np
from Models.BeamSearch import beam_tree
from Data.reward import scores
from Data.data_utils import token_tree_to_id_tree
import pprint


class EncoderDecoder(object):
    def __init__(self, encoder_name, decoder_name, hps, device=None, logdir=None):
        super(EncoderDecoder, self).__init__()

        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self._hps = hps
        self.device = device

        if encoder_name == "LSTM":
            from Models.Encoders import EncoderLSTM
            self.encoder = EncoderLSTM(hps["encoder"], device)
        elif encoder_name == "Tree":
            from Models.Encoders import NaryTreeLSTM
            self.encoder = NaryTreeLSTM(hps["encoder"], device)
            if self.decoder_name == "AttLSTM":
                raise AttributeError("Tree encoder is incompatible with fixed length attention decoder")
        else:
            raise ValueError(encoder_name + ' is not a known encoder')

        if "share_embedding" in hps and hps["share_embedding"]:
            embed_matrix = self.encoder.embed_matrix
        else:
            embed_matrix = None

        if decoder_name == "AttLSTM":
            from Models.Decoders import AttentionDecoderLSTM as Decoder
        elif decoder_name == "Pointer":
            from Models.Decoders import PointerDecoder as Decoder
        else:
            self.N = hps["decoder"]["N"]
            if decoder_name == "NaryTreeDecoder":
                from Models.Decoders import NaryTreeDecoder as Decoder
            elif decoder_name == "AttTreeDecoder":
                from Models.Decoders import AttentionTreeDecoder as Decoder
            elif decoder_name == "PointerTreeDecoder":
                from Models.Decoders import PointerTreeDecoder as Decoder
            elif decoder_name == "ProjAttTreeDecoder":
                from Models.Decoders import ProjAttTreeDecoder as Decoder

            else:
                raise ValueError(decoder_name + ' is not a known decoder')
        self.decoder = Decoder(hps["decoder"], device, embed_matrix)

        self.encoder_optim = optim.Adam(
            self.encoder.parameters(), lr=hps["learning_rate"])
        self.decoder_optim = optim.Adam(
            self.decoder.parameters(), lr=hps["learning_rate"])

        self.__step = 0
        self.logdir = logdir
        print("\n", "-" * 15, "Model Config", "-" * 15)
        print("Encoder:", self.encoder.__class__.__name__)
        print("Decoder:", self.decoder.__class__.__name__)
        pprint.pprint(self._hps)
        num_param = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        num_param += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        if embed_matrix is not None:
            num_param -= embed_matrix.numel()
        print("Num of Params:", num_param)
        print("-" * 40, "\n")

    def save(self, logdir):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        with open(os.path.join(logdir, "config.json"), "w") as fp:
            json.dump(self._hps, fp)
        torch.save(self.encoder.state_dict(), logdir + '/encoder.pt')
        torch.save(self.decoder.state_dict(), logdir + '/decoder.pt')

    def __check_point(self):
        path = self.logdir + "/check_point_step_" + str(self.__step)
        os.makedirs(path)
        self.save(path)

    def load(self, logdir):
        try:
            self.encoder.load_state_dict(torch.load(logdir + "/encoder.pt"))
            self.decoder.load_state_dict(torch.load(logdir + "/decoder.pt"))
        except RuntimeError:
            self.encoder.load_state_dict(torch.load(logdir + "/encoder.pt",
                                                    map_location='cpu'))
            self.decoder.load_state_dict(torch.load(logdir + "/decoder.pt",
                                                    map_location='cpu'))

    def __train(self, loss):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        loss.backward(retain_graph=True)
        res = [d.grad for d in self.decoder.parameters() if d.grad is not None and torch.any(torch.isnan(d.grad))]
        if len(res):
            return
        self.encoder_optim.step()
        self.decoder_optim.step()
        self.__step += 1
        if self.__step % 1000 == 0 and self.logdir is not None:
            self.__check_point()

    def decode_one_step(self, yt, encoder_out, rnn_out, input_seq, hs, get_att=False):
        if self.decoder_name == "AttLSTM":
            att, probs, hs = self.decoder(
                hs, yt, encoder_out)
        if self.decoder_name == "Pointer":
            att, probs, hs = self.decoder(
                hs, yt, rnn_out, input_seq)
        elif self.decoder_name == "NaryTreeDecoder":
            probs, hs = self.decoder(hs, yt)
            att = None
        elif self.decoder_name == "AttTreeDecoder":
            att, probs, hs = self.decoder(hs, yt, encoder_outputs=encoder_out)
        elif self.decoder_name == "ProjAttTreeDecoder":
            att, probs, hs = self.decoder(hs, yt, encoder_hidden_states=rnn_out)
        elif self.decoder_name == "PointerTreeDecoder":
            att, probs, hs = self.decoder(
                hs, yt, encoder_hidden_states=rnn_out, encoder_extend_vocab=input_seq)
        else:
            raise ValueError("Unknown Decoder")
        if self.decoder_name != "AttLSTM":
            # add a trivial value to avoid log(0)
            trivial_v = torch.ones_like(probs) * 1e-16
            probs = probs + trivial_v
        logits = torch.log(probs)
        if get_att:
            return att, probs, logits, hs
        return probs, logits, hs

    def encode(self, input_seq, eos):
        """
        :param eos: end symbol
        :param input_seq: input sequence
        :return :
        encoder_outputs, list of projected output of encoder along input seq, batch_size as second dimension
        rnn_outputs, list of (h, c) pairs of encoder along input seq, batch_size as second dimension
        hidden_states: h, c pair of encoder's last states

        output of shape (seq_len, batch, hidden_size)
        h_n of shape (num_layers * num_directions, batch, hidden_size)
        c_n of shape (num_layers * num_directions, batch, hidden_size)

        input seq: unchanged for sequential lstm, listed seq for tree lstm
        """
        if self.encoder_name == "LSTM":
            encoder_outputs, rnn_outputs, hidden_states = self.encoder(input_seq)
            input_refs = input_seq
        elif self.encoder_name == "Tree":
            batch_size = len(input_seq)
            assert batch_size == 1, "batched training for tree is not recommended"
            if self.decoder_name in ["ProjAttTreeDecoder", "PointerTreeDecoder"]:
                input_len = input_seq[0].size()
            else:
                input_len = self._hps["input_len"]
            encoder_outputs = [[None] * batch_size for _ in range(input_len)]
            rnn_outputs = [[None] * batch_size for _ in range(input_len)]
            hidden_states = [None] * batch_size
            input_refs = [[eos[0][0]] * batch_size for _ in range(input_len)]

            for bid, sample_input in enumerate(input_seq):
                sample_encoder_out, sample_rnn_out, sample_input_ref, (hj, cj) = self.encoder(sample_input)
                for sid in range(len(sample_input_ref)):
                    encoder_outputs[sid][bid] = sample_encoder_out[sid]
                    rnn_outputs[sid][bid] = sample_rnn_out[sid]
                    hidden_states[bid] = (hj, cj)
                    input_refs[sid][bid] = sample_input_ref[sid]

                for pad_id in range(sid, input_len):
                    encoder_outputs[pad_id][bid] = self.encoder.empty_out
                    rnn_outputs[pad_id][bid] = self.encoder.empty_hc

            # concat tensors
            for idx in range(input_len):
                encoder_outputs[idx] = torch.cat(encoder_outputs[idx], dim=0)
                rnn_outputs[idx] = torch.cat(rnn_outputs[idx], dim=0)

            encoder_outputs = torch.stack(encoder_outputs, dim=0)
            rnn_outputs = torch.stack(rnn_outputs, dim=0)

            h_cat = torch.cat([x[0] for x in hidden_states], dim=0)
            h_cat = h_cat.reshape(1, h_cat.shape[0], h_cat.shape[1])
            c_cat = torch.cat([x[1] for x in hidden_states], dim=0)
            c_cat = c_cat.reshape(1, c_cat.shape[0], c_cat.shape[1])
            hidden_states = (h_cat, c_cat)

            input_refs = torch.tensor(input_refs, dtype=torch.long).to(self.device)
        else:
            raise ValueError("Unknown Encoder")
        return encoder_outputs, rnn_outputs, hidden_states, input_refs

    def enforce_encoding(self, *, eos, s1=None, s2=None, encoded1=None, encoded2=None, positive: bool):
        """
        s1 and s2 are equivalent expressions, enforce their encoding to be the same
        i.e. the encoder should encode s1 and s2 to the same vector (encoder output should be same)
        if positive is False, maximize encoding
        """
        assert not (s1 is None and encoded1 is None), "Need tokens or encoding for expression 1"
        assert not (s2 is None and encoded2 is None), "Need tokens or encoding for expression 2"
        encoder_outputs1 = self.encode(s1, eos)[2] if encoded1 is None else encoded1
        encoder_outputs2 = self.encode(s2, eos)[2] if encoded2 is None else encoded2

        loss = torch.nn.MSELoss()(encoder_outputs1[0][:self._hps["hidden_unit_split"]],
                                  encoder_outputs2[0][:self._hps["hidden_unit_split"]]) * self._hps["l2_weight"]
        self.__train(int(positive) * loss)
        return loss.item()

    def supervised(self, input_seq, output_seq, sos, eos=None, train=True):
        """
        :param input_seq:
        :param output_seq:
        :param sos: [1, batch_size], np.array
        :param eos: [1, 1], np.array
        :param train: bool, perform gradient descent or not
        :return: loss
        """
        criterion = torch.nn.NLLLoss()
        encoder_outputs, rnn_outputs, hidden_states, input_seq = self.encode(input_seq, eos)
        if self.decoder_name in ["AttLSTM", "Pointer"]:
            loss = self.__linear_supervised(criterion, sos,
                                            input_seq, output_seq,
                                            encoder_outputs, rnn_outputs,
                                            hidden_states)
        elif self.decoder_name in ["NaryTreeDecoder", "AttTreeDecoder", "ProjAttTreeDecoder", "PointerTreeDecoder"]:
            loss = self.__tree_supervised(criterion, sos, eos,
                                          input_seq, output_seq,
                                          encoder_outputs, rnn_outputs,
                                          hidden_states)

        if train:
            self.__train(loss)

        return loss.item()

    def __linear_supervised(self, criterion, sos, input_seq, output_seq,
                            encoder_outputs, rnn_outputs, hidden_states):
        total_loss = 0
        yt = sos
        for t in range(self._hps["output_len"]):
            _, logits, hidden_states = self.decode_one_step(
                yt, encoder_outputs, rnn_outputs, input_seq, hidden_states)
            logits = logits.reshape([self._hps["batch_size"], -1])
            target_t = torch.from_numpy(output_seq[t]).to(self.device)
            total_loss += criterion(logits, target_t)
            yt = target_t.reshape([1, -1])
        return total_loss

    def __tree_supervised(self, criterion, sos, eos, input_seq, output_seq,
                          encoder_outputs, rnn_outputs, hidden_state):
        """
        assume batched, but tree decoder can not parallel on batch
        :param hidden_state: (h, c), both of size [1, batch_size, hidden_size]
        """
        total_loss = 0
        batch_size = len(output_seq)
        for bid in range(batch_size):
            hs = (hidden_state[0][:, bid, :, ], hidden_state[1][:, bid, :, ])
            queue = [(sos[:, bid], output_seq[bid], hs, 1)]  # BFS
            this_encoder_out = encoder_outputs[:, bid: bid + 1, :, ]
            this_rnn_out = rnn_outputs[:, bid: bid + 1, :, ]
            this_input_seq = input_seq[:, bid: bid + 1]
            while len(queue):
                yt, node, hs, d = queue.pop(0)
                yt = yt.reshape([1, 1])
                probs, logits, hs = self.decode_one_step(
                    yt, this_encoder_out, this_rnn_out, this_input_seq, hs)
                logits = logits.reshape([1, -1])

                if node is None:
                    next_t = eos.reshape([1])
                else:
                    next_t = np.array([node.get_value()])
                target_t = torch.from_numpy(next_t).type(torch.LongTensor).to(self.device)
                total_loss += criterion(logits, target_t)
                if node is None:
                    continue

                cid = 0
                for child in node.iter_child():
                    hsc = (hs[0][cid], hs[1][cid])
                    queue.append((target_t, child, hsc, d + 1))
                    cid += 1

                for ccid in range(cid, len(hs[0])):
                    hsc = (hs[0][ccid], hs[1][ccid])
                    queue.append((target_t, None, hsc, d + 1))
        return total_loss

    def tree_beam_search(self, sample_input, vocab, max_depth, beam_size, num_res, train, bin_score=True,
                         baseline=0, LUT=None):
        if not isinstance(sample_input[0], list):
            # convert to a list
            input_seq = sample_input[0].to_tokens()
        else:
            input_seq = sample_input[0]

        input_seq = vocab.int2str(input_seq)
        equivalence = list()
        unequals = list()
        rl_loss = 0
        total_reward = 0
        total_sample = 0
        for depth in range(1, max_depth + 1):
            bs_results = beam_tree(self, sample_input, vocab, depth, beam_size, num_res, get_attention=False)
            if len(bs_results) == 0:
                continue
            prob_norm = sum(torch.exp(item[1]) for item in bs_results)
            if prob_norm == 0.0:
                prob_norm = bs_results[int(len(bs_results) / 2)][1]
            else:
                prob_norm = torch.log(prob_norm)
            for output_tree, log_prob in bs_results:
                R = scores(output_tree.to_tokens(), input_seq, vocab, bin_score=bin_score, LUT=LUT)
                if 1 - R[2] < 1e-7:
                    equivalence.append(output_tree)
                    rewd = 1.0 * (0.9 ** (output_tree.size()))
                else:
                    rewd = -0.1 / (0.9 ** (output_tree.size()))
                    unequals.append(output_tree)
                total_reward += rewd
                total_sample += 1
                rl_loss -= (rewd - baseline) * (log_prob - prob_norm)
            if len(equivalence):
                break

        if train:
            self.__train(rl_loss)
            if self._hps["l2_loss"]:
                eos = np.ones([1, 1]) * vocab.end()
                _, _, hidden_states, _ = self.encode(sample_input, eos)
                for e in equivalence:
                    e = e.clone()
                    token_tree_to_id_tree(e, vocab)
                    self.enforce_encoding(s2=[e], encoded1=hidden_states, eos=eos, positive=True)
                for ne in unequals:
                    ne = ne.clone()
                    token_tree_to_id_tree(ne, vocab)
                    self.enforce_encoding(s2=[ne], encoded1=hidden_states, eos=eos, positive=False)
        return total_reward / total_sample, rl_loss, equivalence
