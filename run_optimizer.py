from Data.data_utils import token_tree_to_id_tree, id_tree_to_token_tree
from torch.distributions.categorical import Categorical
from Data.reward import _tester
from Data.Batcher import PipelineBatcher, CurriculumBatcher
from Models.train_utils import *
from Models.EncoderDecoder import EncoderDecoder
from Models.Classifier import RewriteDecider
from Optimizer.transform_rule import RuleSet
from Optimizer.data_norm import *


config = "configs/halide.json"
with open(config) as fp:
    hps = json.load(fp)

MAX_ITER = 20
THRESH = 0.1

encoder_name = hps["encoder_name"]
decoder_name = hps["decoder_name"]

pretrain_batcher = CurriculumBatcher(hps["vocab_name"], hps["n_var"], n_sample=hps["n_sample"])
pipeline_batcher = PipelineBatcher()
vocab = pipeline_batcher.vocab
calibrate_hps(hps, vocab, decoder_name)


e2d = EncoderDecoder(encoder_name, decoder_name, hps, device=get_device())
picker = RewriteDecider(hps, device=get_device())

if "load_dir" in hps:
   e2d.load(hps["load_dir"])
   print("Model loaded from", hps["load_dir"])

rule_set = RuleSet(vocab)

if not os.path.exists(hps["logdir"]):
    os.makedirs(hps["logdir"])

batched_sos = [[vocab.start()]]
eos = [[vocab.end()]]
pretrain_saver = RLModelSaver()
pipeline_saver = RLModelSaver()


def pretrain():
    base_r = 0
    for depth, max_epoch in zip([2, 3, 4], [10, 5, 5]):
        for epoch in range(max_epoch):
            reward = 0
            hits = [0, 0, 0]
            totals = [0, 0, 0]

            for idx in range(3):
                params = [0] * 3
                params[idx] = 1
                for tree, _ in pretrain_batcher.level(depth, *params):
                    if epoch == 0:
                        token_tree_to_id_tree(tree, vocab)
                    r, loss, equals = e2d.tree_beam_search(
                        [tree], vocab,
                        max_depth=tree.depth(),
                        beam_size=20, num_res=20, train=params[0], baseline=base_r)
                    if len(equals):
                        hits[idx] += 1
                        rhs = sorted(equals, key=lambda x: str(x.size()) + str(x))[0]
                        rule_set[tree] = rhs
                    print(f"{tree} --> {equals}, with loss {loss}")
                    totals[idx] += 1
                    reward += r

            print(f"[SUM] ==== Epoch {epoch} Depth {depth}  ===========")
            print(f"[SUM] Train      {hits[0]} / {totals[0]}")
            print(f"[SUM] Testing    {hits[1]}  / {totals[1]}")
            print(f"[SUM] Validation {hits[2]} / {totals[2]}\n")

            pretrain_saver.save(e2d, "pretrain_log", depth, *hits, rule_set)
            base_r = reward / sum(totals)


def pipeline_train(e2d, picker, log_prob, picker_reward, train):
    if not train:
        return
    e2d.encoder_optim.zero_grad()
    picker.optim.zero_grad()
    loss = -log_prob * picker_reward
    loss.backward(retain_graph=True)
    e2d.encoder_optim.step()
    picker.optim.step()


def rule_rewrite(rule_set, sub_tree, root):
    id_tree_to_token_tree(sub_tree, vocab)
    matching = rule_set.partially_match_rule(sub_tree)
    if matching:
        new_subtree = matching.partially_transform(sub_tree)
        if id(sub_tree) == id(root):
            tree = new_subtree
        else:
            replace(sub_tree, new_subtree)
            tree = root
        picker_reward = 10.0
        return tree, picker_reward

    # test soft match
    exp_soft_rewrite = rule_set.soft_rewrite(sub_tree)
    str_re = str(exp_soft_rewrite)
    str_tr = str(sub_tree)
    if str_re != str_tr and "(" + str_re + ")" != str_tr:
        # rewrite succeed
        if id(sub_tree) == id(root):
            tree = exp_soft_rewrite
        else:
            replace(sub_tree, exp_soft_rewrite)
            tree = root
        picker_reward = 10.0
        return tree, picker_reward
    # not matched
    return root, -1


def reduce_const(tree, vocab, LUT):
    rename(tree, LUT)
    tree.reduce(vocab)

    LUT = dict()
    for nid, node in enumerate(tree.get_postorder_list()):
        if node is None:
            continue
        const_value = vocab.to_const(node.get_value())
        if const_value is not None:
            val = "c" + str(len(LUT))
            for v in LUT:
                if LUT[v] == const_value:
                    val = v
            LUT[val] = const_value
            node.set_value(val)
    return tree, LUT


def pipeline(tree, LUT, train, base_r):
    n_iter = 0
    # perform reduce
    tree, LUT = reduce_const(tree, vocab, LUT)
    if tree.size() == 1:
        return tree, LUT

    while n_iter < MAX_ITER:
        base_length = len(tree.to_tokens())
        id_tree = tree.clone()
        token_tree_to_id_tree(id_tree, vocab)
        encoder_out, rnn_out, hidden_states, input_refs = e2d.encode([id_tree], eos)
        select_scores = picker(encoder_out)
        if torch.max(select_scores) < THRESH:
            break
        disp = Categorical(probs=select_scores)
        sub_tree_id = disp.sample()
        log_prob = disp.log_prob(sub_tree_id)
        sub_tree = id_tree.get_postorder_list()[sub_tree_id]
        n_iter += 1

        if sub_tree.get_type() == VAR:
            pipeline_train(e2d, picker, log_prob, picker_reward=-1, train=train)
            continue

        tree, picker_reward = rule_rewrite(rule_set, sub_tree, id_tree)
        if picker_reward < 0:
            sub_tree, cef_LUT = cef(sub_tree, vocab)
            sub_tree, SEE = sef(sub_tree, 3, vocab)
            sub_tree, sub_LUT = normalize(sub_tree, vocab)
            token_tree_to_id_tree(sub_tree, vocab)

            merged_lut = {}
            for k, v in sub_LUT.items():
                if k in LUT and k not in SEE:
                    merged_lut[v] = LUT[k]
            _, loss, equals = e2d.tree_beam_search(
                [sub_tree], vocab,
                max_depth=min(3, sub_tree.depth()),
                beam_size=20, num_res=20, train=train,
                baseline=base_r, LUT=merged_lut)
            id_tree_to_token_tree(sub_tree, vocab)
            if len(equals):
                reprs = [str(x) for x in equals]
                if 'True' in reprs:
                    rhs = TreeNode(val='True', t=VAR)
                elif 'False' in reprs:
                    rhs = TreeNode(val='False', t=VAR)
                else:
                    rhs = sorted(equals, key=lambda x: str(x.size()) + str(x))[0]
                id_tree_to_token_tree(sub_tree, vocab)
                if not any([i in merged_lut for i in sub_tree.to_tokens()]):
                    rule_set[sub_tree] = rhs
                else:
                    local_equiv_test = _tester(sub_tree, rhs, vocab, binary=True)
                    if local_equiv_test:
                        rule_set[sub_tree] = rhs
                picker_reward = 5.0
                denormalize(rhs, sub_LUT)
                denormalize(sub_tree, sub_LUT)
                rhs = revs_sef(rhs, SEE)
                sub_tree = revs_sef(sub_tree, SEE)
                token_tree_to_id_tree(sub_tree, vocab)
                token_tree_to_id_tree(rhs, vocab)

                if sub_tree.get_parent() is None:
                    id_tree = rhs
                else:
                    replace(sub_tree, rhs)
                id_tree_to_token_tree(id_tree, vocab)
                token_tree_to_id_tree(id_tree, vocab)
                tree = id_tree
            else:
                denormalize(sub_tree, sub_LUT)
                revs_sef(sub_tree, SEE)
                picker_reward = -1
                tree = id_tree
            revs_cef(tree, cef_LUT)

        id_tree_to_token_tree(tree, vocab)
        tree, LUT = reduce_const(tree, vocab, LUT)
        if tree.size() == 1:
            break

        if picker_reward > 0:
            picker_reward *= base_length - len(tree.to_tokens())

        pipeline_train(e2d, picker, log_prob, picker_reward, train)

    return tree, LUT


baseline_reward = 0
pretrain()
for epoch in range(1):
    rewards = 0
    size_red = [0, 0, 0]
    length_red = [0, 0, 0]
    clen_red = [0, 0, 0]
    iterators = [pipeline_batcher.next_train(), pipeline_batcher.next_test(), pipeline_batcher.next_val()]
    pools = [pipeline_batcher.train_set, pipeline_batcher.test_set, pipeline_batcher.val_set]
    for idx, iterator, is_train, pool in zip([0, 1, 2], iterators, [True, False, False], pools):
        for tree, LUT in iterator:
            new_tree, new_LUT = pipeline(tree.clone(), LUT, is_train, baseline_reward)
            print("Length reduction", len(str(tree)) - len(str(new_tree)))
            print("Size reduction", tree.size() - new_tree.size())
            rename(tree, LUT)
            rename(new_tree, new_LUT)
            print(f"{tree} --> {new_tree}")
            size_red[idx] += tree.size() - new_tree.size()
            length_red[idx] += len(tree.to_tokens()) - len(new_tree.to_tokens())
            clen_red[idx] += len(str(tree)) - len(str(new_tree))
        size_red[idx] /= len(pools)
        length_red[idx] /= len(pools)

    print(f'''[SUM] Pipeline Epoch {epoch}\n[SUM] Size Reduction {size_red}\n[SUM] Length Reduction {length_red}\n''')
    pipeline_saver.save(e2d, "train_pipeline_log", 10, *size_red)
