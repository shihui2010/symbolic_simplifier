from Data.data_utils import TreeNode, Vocab, VAR


def normalize(exp: TreeNode, vocab: Vocab) -> (TreeNode, dict):
    """
    the method (inplace) renames variables in the expression according to the order they appear.
    :param exp: the expression, as tree
    :param vocab: the vocabulary, all variables present in expression should be in the vocab
    :return: normalized expression, loop up table as dict {origin name -> new name}
    """
    LUT = dict()
    var_list = vocab.get_vars()
    if vocab.is_var(exp.get_value()):
        LUT[exp.get_value()] = var_list[0]
        exp.set_value(var_list[0])
        return exp, LUT

    stack = [exp]
    while len(stack):
        p = stack.pop()
        if vocab.is_var(p.get_value()):
            if p.get_value() not in LUT:
                LUT[p.get_value()] = var_list[len(LUT)]
            p.set_value(LUT[p.get_value()])

        cache = list()
        for c in p.iter_child():
            if c is not None:
                cache.append(c)
        # reverse order
        for c in cache[::-1]:
            stack.append(c)
    return exp, LUT


def replace(tree: TreeNode, new_tree: TreeNode):
    """Assume tree should have a parent"""
    par = tree.get_parent()
    for cid, c in enumerate(par.iter_child()):
        if c is None:
            continue
        if id(c) == id(tree):
            new_tree.free_parent()
            par.set_child(cid, new_tree)
            tree.free_parent()
    return par


def sef(exp: TreeNode, depth: int, vocab: Vocab) -> (TreeNode, dict):
    """
    Sub-Expression Folding. replace sub-tree with single node, such that the depth of root
    matches depth.
    :return:
    """
    cut_off = exp.depth() - depth + 1
    if cut_off == 1:
        return exp, {}
    unused_vars = [v for v in vocab.word2id if v not in exp.to_tokens() and vocab.is_var(v)]
    LUT = dict()
    nodes = exp.get_postorder_list()
    for n in nodes:
        if n.depth() == cut_off:
            for key in LUT:
                if str(key) == str(n):
                    pass
            tmp_v = unused_vars[len(LUT)]
            LUT[tmp_v] = n
    for tmp_v in LUT:
        tmp_n = TreeNode(val=tmp_v, t=VAR)
        replace(LUT[tmp_v], tmp_n)
    return exp, LUT


def cef(exp: TreeNode, vocab: Vocab) -> (TreeNode, dict):
    """common sub-expression folding"""
    nodes = exp.get_postorder_list()
    unused_vars = [v for v in vocab.word2id if v not in exp.to_tokens() and vocab.is_var(v)]
    LUT = dict()  # {str: TreeNode}
    dup = dict()
    for r in nodes:
        if r.size() > 1:
            rep = str(r)
            dup.setdefault(rep, list()).append(r)
    for string in sorted(dup.keys(), key=lambda x: -len(x)):
        if len(dup[string]) > 1:
            if string not in str(exp):
                continue
            tmp_v = unused_vars[len(LUT)]
            for n in dup[string]:
                tmp_n = TreeNode(val=tmp_v, t=VAR)
                replace(n, tmp_n)
            LUT[tmp_v] = dup[string][0].clone()
    return exp, LUT


def revs_cef(exp: TreeNode, LUT: dict) -> TreeNode:
    """reverse process of common sub-expression folding"""
    queue = [exp]
    while len(queue):
        node = queue.pop(0)
        if node.size() == 1 and str(node) in LUT:
            replace(node, LUT[str(node)].clone())
        elif node.size() > 1:
            for c in node.iter_child():
                if c is not None:
                    queue.append(c)
    return exp


def revs_sef(exp: TreeNode, LUT: dict) -> TreeNode:
    res = exp
    if not len(LUT):
        return res
    nodes = exp.get_postorder_list()
    for n in nodes:
        if n.get_value() in LUT:
            if n.get_parent() is None:
                res = LUT[n.get_value()]
            else:
                replace(n, LUT[n.get_value()])
    return res


def denormalize(exp: TreeNode, LUT: dict) -> TreeNode:
    """
    The method performs (in place) inverse normalize to the expression.
    :param exp: expression as tree node
    :param LUT: LookUpTable for variable renaming
    :return: denormalized tree
    """
    inv_LUT = {v:k for k, v in LUT.items()}
    rename(exp, inv_LUT)
    return exp


def rename(root: TreeNode, LUT: dict) -> None:
    """
    inplace renaming according to LUT
    """
    if root.get_child_num() == 0:
        if root.get_value() in LUT:
            root.set_value(str(LUT[root.get_value()]))
        return
    for c in root.iter_child():
        if c is not None:
            rename(c, LUT)

