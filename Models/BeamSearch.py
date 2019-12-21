import numpy as np
import torch
from itertools import product
from functools import reduce
from operator import add
from Data.data_utils import new_node_by_val


def iter_input(input_seq):
    if isinstance(input_seq, list):
        for s in input_seq:
            yield [s]
    else:
        batch_size = input_seq.shape[1]
        for bid in range(batch_size):
            yield np.take(input_seq, [bid], axis=1)


def beam_search(model, sample_input, vocab, max_len, beam_size=10):
    """
    not batched
    :param vocab: Vocab object
    :param max_len: max decode length
    :return: vocab_ids
    """
    vids = np.zeros([max_len, beam_size], dtype=int)
    # probs = np.zeros([max_len, beam_size], dtype=float)
    probs = [[0] * beam_size for _ in range(max_len)]
    back = np.zeros([max_len, beam_size], dtype=int)
    # back tracks beam id, not vocab id

    # encode
    eos = np.ones([1, 1]) * vocab.end()
    encoder_out, rnn_out, hidden_state, sample_input = model.encode(sample_input, eos)

    # step1
    sos = np.ones([1, 1]) * vocab.start()
    _, logits, s = model.decode_one_step(sos, encoder_out, rnn_out, sample_input, hidden_state)
    decoder_states = [s] * beam_size
    topk_p, topk_idx = logits.topk(beam_size, dim=2)
    probs[0] = topk_p[0, 0]  # .cpu().data
    back[0] += vocab.start()
    vids[0] = topk_idx[0, 0].cpu().data

    for t in range(max_len - 1):
        hyps = dict()  # (vid: [accum_log_prob, back, states])
        for i in range(beam_size):
            base_prob = probs[t][i]
            last_states = decoder_states[i]
            yt = vids[t, i].reshape([1, 1])
            _, logits, s = model.decode_one_step(yt, encoder_out, rnn_out, sample_input, last_states)
            topk_p, topk_idx = logits.topk(beam_size, dim=2)
            for j in range(beam_size):
                accum_p = topk_p[0, 0, j] + base_prob
                v = topk_idx[0, 0, j].item()
                if v not in hyps or hyps[v][0] < accum_p:
                    hyps[v] = (accum_p, i, s)

        # pruning
        topk_hyps = sorted(hyps.keys(), key=lambda x: -hyps[x][0])
        for i in range(beam_size):
            probs[t + 1][i] = hyps[topk_hyps[i]][0]
            vids[t + 1, i] = topk_hyps[i]
            back[t + 1, i] = hyps[topk_hyps[i]][1]
            decoder_states[i] = hyps[topk_hyps[i]][2]

    # back track
    sequence = np.zeros([max_len], dtype=int)
    back_id = np.argmax([i for i in probs[-1]])
    best_prob = probs[-1][back_id]
    sequence[-1] = vids[-1, back_id]
    for t in range(1, max_len):
        back_id = back[-t, back_id]
        sequence[-t - 1] = vids[-t - 1, back_id]
    return [vocab.id2w(i) for i in sequence], best_prob  # np.max(probs[-1])


def masked_topk(logits, vocab, k, dim, functional=False, var=True, op=True, nullable=False):
    """
    assumes dimension of logits includes </s>, not includes <pad>, (, ), ","
    only use </s> for empty child for special cases
    """
    topk_p, topk_idx = logits.topk(logits.shape[dim], dim=dim)
    new_top_p, new_top_idx = list(), list()
    for p, i in zip(topk_p[0, 0], topk_idx[0, 0]):
        wid = i.view(1).item()
        if not functional and wid < 6:
            if not (nullable and wid == 2):
                continue
        if not var and vocab.is_const_or_var(wid):
            continue
        if not op and (vocab.is_op(wid) or vocab.is_func(wid)):
            continue
        p = p.view(1)  # .item()
        new_top_p.append(p)
        new_top_idx.append(i)
        if len(new_top_p) == k:
            break
    return new_top_p, new_top_idx


def beam_tree(model, sample_input, vocab, max_depth, beam_size=10, num_res=10, get_attention=False):
    """
    not batched, tree decoder
    :param vocab: Vocab object
    :param max_depth: max decode depth
    :return:
    """
    DEBUG = False

    def print_masked(tokens, msg):
        if not DEBUG:
            return
        res = list()
        for i in tokens:
            res.append(vocab.id2w(i.view(1).item()))
        print(msg, res)
        input()

    # encode
    eos = np.ones([1, 1]) * vocab.end()
    encoder_out, rnn_out, hidden_state, sample_input = model.encode(sample_input, eos)

    # track structure
    # len(track_by_layer) = # of layers
    # for each layer, the list consist of beam_size of tuples as:
    # ([m tokens in the same layer], [m list of hidden state for next layer], base_prob, back_id)
    # back_id points to the idx in the list of last layer, not vid

    track_by_layer = [[]]

    # step1, decode root node
    sos = np.ones([1, 1]) * vocab.start()
    _, logits, s = model.decode_one_step(sos, encoder_out, rnn_out, sample_input, hidden_state)
    op = max_depth != 1
    topk_p, topk_idx = masked_topk(logits, vocab, beam_size, op=op, dim=2, nullable=False)
    print_masked(topk_idx, "masked functional token")

    for p, i in zip(topk_p, topk_idx):
        if DEBUG:
            print("enqueue first layer")
            print("i", [i])
            print("p", p)
            print("s: obj ids", id(s), [id(j) for j in s[0]], [id(j) for j in s[1]])
            print("-" * 10, "\n")
            input()
        track_by_layer[0].append(([i], [s], p, -1))

    while len(track_by_layer) <= max_depth:
        hyps = dict()
        # the hash key is the tuple of instantiation of all node in this layer
        # the value is (accum prob, back_id, (tuple of hidden state))

        for back_id, (yt_list, hs_list, base_prob, _) in enumerate(track_by_layer[-1]):
            if DEBUG:
                print("back_id", back_id)
                print("yt_list", yt_list, type(yt_list))
                print("hs_list", len(hs_list[0]), len(hs_list[0][0]), hs_list[0][0][1].shape)
                print("base_prob", base_prob)
                print("s states", id(hs_list[0]))
                # input()

            reduced_CP_by_branch = list()
            # list of instantiation of child nodes for each branch (i.e., yt)
            # this instantiation list are upto length of beam_size (i.e., reduced)
            # len(reduced_CP_by_branch = number of branches from based on last layer
            # len(reduced_CP_by_branch[0] = beam_size (top combinations for current branch)
            # reduced_CP_by_branch[0][0] = ((p1, token1), (p2, token2),...(p_num_operands, token_num_operands))

            next_chs = list()
            # whatever the options for child node of each branch, they shares next hidden states
            # also the back_id is shared

            # for each branch <yt, hs>, decode its children
            for yt, hs in zip(yt_list, hs_list):
                val = yt.view(1).item()
                yt = yt.view(1, 1)

                options_by_child = list()
                # stores num_operands list, each of which contains beam-size options for child node
                # next_chs = list()

                # if yt is None, or terminate node, decode None for all children
                if val == vocab.end() or vocab.is_const_or_var(val):
                    term_prob = 0.0
                    for cid in range(model.N):
                        _, c_probs, c_hs = model.decode_one_step(yt, encoder_out, rnn_out, sample_input,
                                                                 (hs[0][cid], hs[1][cid]))
                        next_chs.append(c_hs)
                        term_prob += c_probs[0, 0, vocab.end()].view(1)  # .item()
                    reduced_CP_by_branch.append(
                        ([([torch.tensor([vocab.end()]) for _ in range(model.N)], term_prob)]))

                    continue

                # reduced_CP = list()
                # # Do Cartesian product for beam_size options for each child node of current branch
                # # then reduce the outcome to top beam_size products
                # # reduced_cp stores list of ((c_1, c_2, ..), (hs_1, hs_2, ...), overall_prob, back_id)
                # # the length of list is less than beam size

                # if yt is operator, get number of operands
                num_operands = vocab.num_operands(val)
                if DEBUG:
                    print("val", vocab.id2w(val), "num_operands", num_operands)

                for cid in range(num_operands):
                    _, c_probs, c_hs = model.decode_one_step(yt, encoder_out, rnn_out, sample_input,
                                                             (hs[0][cid], hs[1][cid]))

                    nullable = cid == 0 and val == vocab.w2id('-')
                    if len(track_by_layer) == max_depth - 1:
                        topk_p, topk_idx = masked_topk(c_probs, vocab, k=beam_size, dim=2, op=False, nullable=nullable)
                        print_masked(topk_idx, "masking to have only vars")
                    else:
                        topk_p, topk_idx = masked_topk(c_probs, vocab, k=beam_size, dim=2, nullable=nullable)
                        print_masked(topk_idx, "masked functional tokens")

                    options_by_child.append([(i, p) for p, i in zip(topk_p, topk_idx)])
                    next_chs.append(c_hs)
                    # back_id and next_hidden_states are shared

                for cid in range(num_operands, model.N):
                    _, c_probs, c_hs = model.decode_one_step(yt, encoder_out, rnn_out, sample_input,
                                                             (hs[0][cid], hs[1][cid]))
                    term_prob = c_probs[0, 0, vocab.end()].view(1)  # .item()
                    # term_prob = 1.0
                    options_by_child.append([(torch.tensor([vocab.end()]), term_prob)])
                    next_chs.append(c_hs)

                # Do Cartesian product on options_by_child. Keep only top results
                CP_pool = list()  # keep this being sorted, as beam size is small
                if DEBUG:
                    print("options_by_child", len(options_by_child))
                for c_states in product(*options_by_child):
                    if DEBUG:
                        print("c_states", c_states)
                    overall_prob = reduce(add, [c[1] for c in c_states])
                    if len(CP_pool) < beam_size:
                        CP_pool.append(([[c[0] for c in c_states], overall_prob]))
                    elif overall_prob > CP_pool[-1][1]:  # better than worst one
                        CP_pool.pop()
                        CP_pool.append(([[c[0] for c in c_states], overall_prob]))
                        CP_pool = sorted(CP_pool, key=lambda x: -x[1])

                reduced_CP_by_branch.append(CP_pool)
            if DEBUG:
                print("length of c_hs", len(next_chs))
                # input()
                print("reduced_CP_by_branch", len(reduced_CP_by_branch), len(reduced_CP_by_branch[0]))
                for cp in reduced_CP_by_branch:
                    print(cp)
                print("-" * 20)

            # Do Cartesian product by branch. Keep only top results
            branch_CPs = list()
            for b_states in product(*reduced_CP_by_branch):
                if DEBUG:
                    print("branch states", type(b_states), len(b_states), b_states)
                overall_prob = reduce(add, [b[1] for b in b_states]) + base_prob

                if len(branch_CPs) < beam_size:
                    branch_CPs.append(([b[0] for b in b_states], overall_prob))
                elif overall_prob > branch_CPs[-1][1]:
                    branch_CPs.pop()
                    branch_CPs.append(([b[0] for b in b_states], overall_prob))
                    branch_CPs = sorted(branch_CPs, key=lambda x: -x[1])

            # hash to hypothesis dict
            for cartesian in branch_CPs:
                # WARNING: tuple of torch tensor is not hashed correctly
                # keep using tuple of ints as key
                s = list()
                for i in cartesian[0]:
                    s.extend(i)
                key = tuple(i.view(1).item() for i in s)
                if DEBUG:
                    print("cartesian", cartesian)
                    print("s", s, "key:", key)
                if key in hyps:
                    if hyps[key][1] < cartesian[1]:
                        hyps[key] = (s, cartesian[1], back_id, next_chs)
                else:
                    hyps[key] = (s, cartesian[1], back_id, next_chs)

            if DEBUG:
                print("length of hypothesis", len(hyps))
                for k in hyps:
                    print("hypothesis keys", k)

        # prune hypothesis
        topk_hyps = sorted(hyps.keys(), key=lambda x: -hyps[x][1])
        this_layer = list()
        for i in range(min(beam_size, len(topk_hyps))):
            key = topk_hyps[i]
            s = hyps[key][0]
            accum_p = hyps[key][1]
            back_id = hyps[key][2]
            hs_list = hyps[key][3]
            this_layer.append((list(s), hs_list, accum_p, back_id))

        # move to next layer
        track_by_layer.append(this_layer)

        if DEBUG:
            print("this_layer", len(this_layer))  # , this_layer[0]

        if len(this_layer) == 1:
            if __terminates(this_layer[0][0], vocab):
                break

        # END expanding layer

    for_print = list()
    for layer in track_by_layer:
        for_print.append([])
        for s in layer:
            for_print[-1].append((s[0], s[2], s[3]))

    if num_res == 1:
        return _back_track(track_by_layer, vocab, model, 0, DEBUG)

    if len(track_by_layer[-1]) == 1 and len(track_by_layer) > 1:
        track_by_layer = track_by_layer[:-1]

    res = list()
    for i in range(min(num_res, len(track_by_layer[-1]))):
        try:
            root, prob = _back_track(track_by_layer, vocab, model, i, DEBUG)
            if root.depth() == max_depth:
                res.append(_back_track(track_by_layer, vocab, model, i, DEBUG))
        except IndexError:
            break
    return res


def __terminates(states, vocab):
    for wid in states[0]:
        if wid.view(1).item() != vocab.end():
            return False
    return True


def _back_track(track_by_layer, vocab, model, top_id, DEBUG=False):
    """
    perform back tracking for tree beam search
    :param track_by_layer: created in tree beam
    :param vocab: vocab
    :param model: encoder-decoder model
    :param top_id: the tree with i-th large probability
    :param DEBUG: bool, print immediate information
    :return: tree and the probability
    """
    # back track
    rev_seq = list()
    if DEBUG:
        print("Top id", top_id)

    back_id_entry = sorted(range(len(track_by_layer[-1])), key=lambda x: -track_by_layer[-1][x][2])[top_id]
    max_prob = track_by_layer[-1][back_id_entry][2]  # prob of best sample
    back_id = track_by_layer[-1][back_id_entry][3]
    if DEBUG:
        print("back_id", back_id, "max_prob", max_prob)
    rev_seq.append(track_by_layer[-1][back_id_entry][0])  # keep tokens only
    for t in range(1, len(track_by_layer)):
        rev_seq.append(track_by_layer[-t - 1][back_id][0])
        back_id = track_by_layer[-t - 1][back_id][3]
        if DEBUG:
            print("back_id", back_id)
    seq = rev_seq[::-1]

    if DEBUG:
        print("seq")
        for s in seq:
            print(s)
        # input()
    # build tree from layers
    root = new_node_by_val(vocab, seq[0][0].view(1).item())
    queue = [root]

    if DEBUG:
        print("\n\ncreated root", root.get_value())
    for lid in range(1, len(seq)):
        new_queue = list()
        sid = 0
        if DEBUG:
            print("new layer")
            print("lid=", lid, "length of seq[lid]", len(seq[lid]), "seq:", seq[lid])
        while sid != len(seq[lid]):
            if sid == 0:
                this_root = queue.pop(0)
            if this_root is None:
                # empty child, append N nones to next layer
                new_queue.extend([None] * model.N)
            else:
                num_operands = vocab.num_operands(this_root.get_value())
                if num_operands == 0:
                    # variable/constant, append N nones to next layer
                    new_queue.extend([None] * model.N)
                else:
                    if DEBUG:
                        print("sid", sid, this_root.get_value(), "have num_op", num_operands)
                    for offset in range(num_operands):
                        if DEBUG:
                            print("\nthis root", this_root.get_value())
                            print("seq[lid][sid + offset]", seq[lid][sid + offset])
                        child = new_node_by_val(vocab, seq[lid][sid + offset].view(1).item())
                        this_root.add_child(child)
                        new_queue.append(child)

                    for offset in range(num_operands, model.N):
                        new_queue.append(None)

            sid += model.N

            if len(queue):
                this_root = queue.pop(0)
        queue = new_queue
        if DEBUG:
            print(root)
            # input()
    return root, max_prob
