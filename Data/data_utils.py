import numpy as np
try:
    from graphviz import Graph
    GUI = True
except ImportError:
    GUI = False

FUNC = "func"
OP = "op"
VAR = "var"
TYPES = [FUNC, OP, VAR]
color_bin = np.array(np.arange(8) / 80)
color_bin[-1] = 1
color_map = ['white', 'lightblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'royalblue', 'blue', 'navy']


def new_node_by_val(vocab, val):
    if isinstance(val, int):
        val = vocab.id2w(val)
    elif not isinstance(val, str):
        raise TypeError("Type of val is" + str(type(val)))
    if vocab.is_const_or_var(val):
        t = VAR
    elif vocab.is_func(val):
        t = FUNC
    elif vocab.is_op(val):
        t = OP
    else:
        return None
    return TreeNode(val=val, t=t)


class Vocab(object):
    def __init__(self):
        self.word2id = {"<s>": 0, "<pad>": 1, "</s>": 2, "(": 3, ")": 4,
                        ",": 5, "0": 6, "1": 7, "2": 8,
                        "True": 9, "False": 10}
        self.op_num = None  # number of operations

    def num_operands(self, token):
        raise NotImplementedError("num_operands not implemented")

    def random_func(self):
        if self.op_num is None:
            raise NotImplementedError("Number of Operation is not set")
        if hasattr(self, "_probs"):
            rid = np.random.choice(self.op_num, size=1, p=self._probs)[0]
        else:
            rid = np.random.randint(0, self.op_num)
        return self.id2word[rid + 11]

    def random_const_and_var(self):
        """
        :return:  a random token being either constant or variable
        """
        if self.op_num is None:
            raise NotImplementedError("Number of Operation is not set")
        const = np.random.rand(1)
        if const > 0.5:  # bias toward variables
            rid = np.random.randint(6, 11)
            return self.id2w(rid)
        rid = np.random.randint(11 + self.op_num, len(self.word2id))
        return self.id2w(rid)

    def _auto_id2word(self):
        self.id2word = {v: k for k, v in self.word2id.items()}

    def add_word(self, token):
        if token not in self.word2id:
            self.word2id[token] = len(self.word2id)
            self.id2word[len(self.id2word)] = token
        return self.word2id[token]

    def get_const_vars(self):
        v = list()
        for w in self.word2id:
            if self.is_const_or_var(w):
                v.append(w)
        return v

    def get_vars(self):
        v = list()
        for w in self.word2id:
            if self.is_var(w):
                v.append(w)
        return v

    def get_ops(self):
        """ include functions and operators"""
        raise NotImplementedError("subclass have not implemented get_ops")

    def id2w(self, idx):
        if isinstance(idx, str):
            return idx
        if isinstance(idx, TreeNode):
            idx = idx.get_value()
            return self.id2w(idx)
        return self.id2word[int(idx)]

    def w2id(self, w):
        return self.word2id[w]

    def start(self):
        return self.word2id["<s>"]

    def end(self):
        return self.word2id["</s>"]

    def pad(self):
        return self.word2id["<pad>"]

    def is_op(self, token):
        """
        op syntax: arg1 op arg2
        """
        raise NotImplementedError("subclass have not implemented is_op")

    def is_func(self, token):
        """
        function syntax: func(arg1, arg2, ...)
        """
        raise NotImplementedError("subclass have not implemented is_func")

    def is_const(self, token):
        return self.to_const(token) is not None

    def to_const(self, token):
        if isinstance(token, int):
            if token not in self.id2word:
                return None
            token = self.id2word[token]
        try:
            if "." in token:
                res = float(token)
            else:
                res = int(token)
            return res
        except ValueError:
            if token.lower() == "true":
                return True
            if token.lower() == "false":
                return False
        return None

    def is_const_or_var(self, token):
        if type(token) == int:
            if token not in self.id2word:
                return False
            token = self.id2word[token]
        if self.is_const(token):
            return True
        if token not in self.word2id:
            return False
        if self.is_op(token):
            return False
        if self.word2id[token] < 6:
            # functional tokens
            return False
        return True

    def is_var(self, token):
        return self.is_const_or_var(token) and not self.is_const(token)

    def str2int(self, tokens):
        res = list()
        for i in range(len(tokens)):
            res.append(self.word2id[tokens[i]] if isinstance(tokens[i], str) else tokens[i])
        return res

    def int2str(self, tokens):
        res = list()
        for i in range(len(tokens)):
            res.append(self.id2word[tokens[i]] if isinstance(tokens[i], int) else tokens[i])
        return res

    def execute(self, op, *args):
        raise NotImplementedError("subclass have not implemented execute")

    def __len__(self):
        return len(self.word2id)

    def __getitem__(self, token):
        return self.word2id[token]


def tokenize(exp, vocab):
    tokens = list()
    start = 0
    end = 1
    while end <= len(exp):
        # print exp[start: end], tokens
        if vocab.is_func(exp[start:end]):
            # print "full length", exp[start: end]
            tokens.append(exp[start:end])
            start = end
            end = start + 1
        elif exp[end - 1] == " ":
            if start < end - 1:
                tokens.append(exp[start: end - 1])
            start = end
            end = start + 1
        elif vocab.is_op(exp[end - 2: end]):
            # print "double condition", exp[end - 2: end]
            if start < end - 2:
                tokens.append(exp[start: end - 2])
            tokens.append(exp[end - 2: end])
            start = end
            end = start + 1
        elif exp[end - 1] in [")", "(", ","] or vocab.is_op(exp[end - 1]):
            # print "condition", exp[end - 1], exp[end - 1: end + 1],
            # print vocab.is_op(exp[end - 1: end + 1])
            if end < len(exp) and vocab.is_op(exp[end - 1: end + 1]):
                end += 1
                continue
            if start < end - 1:
                tokens.append(exp[start: end - 1])
            tokens.append(exp[end - 1])
            start = end
            end = start + 1
        else:
            end += 1
    if start < len(exp):
        tokens.append(exp[start:])
    return tokens


def truncate(tokens, vocab):
    """
    truncate sequence to valid length, i.e. up to </s> symbol
    return the tokens if tokens is treeNode
    """
    if type(tokens) == TreeNode:
        return tokens.to_tokens()
    if len(tokens) == 0:
        return tokens
    valid_tokens = []
    if isinstance(tokens[0], str):
        eos = vocab.id2w(vocab.end())
        for t in tokens:
            if t != eos:
                valid_tokens.append(t)
            else:
                break
    else:
        eos = vocab.end()
        for t in tokens:
            if t != eos:
                valid_tokens.append(vocab.id2w(t))
            else:
                break
    return valid_tokens


class TreeNode(object):
    def __init__(self, val=None, t=None):
        self._val = val
        self._parent = None
        self._children = list()
        self._type = t

    def set_value(self, val):
        self._val = val

    def get_value(self):
        return self._val

    def set_type(self, t):
        assert t in [FUNC, OP, VAR]
        self._type = t

    def get_type(self):
        return self._type

    def free_parent(self):
        self._parent = None

    def add_child(self, node):
        # first child might be none for "-" operation
        self._children.append(node)
        if node is None:
            return
        node.set_parent(self)
        # print "set parent", node._parent

    def set_parent(self, node):
        self._parent = node

    def get_parent(self):
        return self._parent

    def set_child(self, idx, node):
        self._children[idx] = node
        if node is not None:
            node.set_parent(self)

    def iter_child(self):
        for c in self._children:
            yield c

    def get_child(self, idx):
        return self._children[idx]

    def get_child_num(self):
        return len(self._children)

    def __contains__(self, item):
        if item == self._val:
            return True
        for c in self._children:
            if c is not None and item in c:
                return True
        return False

    def execute(self, vocab, feed_dict=None):
        # print "executing: ", str(self)
        if self._type == VAR:
            const_value = vocab.to_const(self._val)
            if const_value is not None:
                return const_value
            else:
                if self._val in feed_dict:
                    return feed_dict[self._val]
                raise KeyError(self._val + " not in feed_dict")
        else:
            if self._val == "-" and self._children[0] is None:
                return - self._children[1].execute(vocab, feed_dict)
            args = [c.execute(vocab, feed_dict) for c in self._children]
            return vocab.execute(self._val, *args)

    def get_const_var_set(self):
        res = set()
        if self._type == VAR:
            res.add(self._val)
        else:
            for c in self._children:
                if c is not None:
                    res.update(c.get_const_var_set())
        return res

    def __repr__(self):
        if self._val is None or self._type is None:
            v = ""  # for debug "<none>"
        else:
            v = str(self._val)
        if self._type == VAR or len(self._children) == 0:
            res = str(v)
        elif self._type == OP:
            if len(self._children) == 1:
                res = v + " " + str(self._children[0])
            else:
                if self._val == '-' and self._children[0] is None:
                    c0 = ""
                else:
                    c0 = str(self._children[0]) + " "
                res = c0 + v + " " + str(self._children[1])
            if self._parent is not None and self._parent.get_type() != FUNC:
                res = "(" + res + ")"
        else:
            # if self._type == _FUNC:
            res = v + "(" + ", ".join(
                str(i) for i in self._children) + ")"
        return res

    def size(self):
        """
        size is total number of nodes in the tree, the root of which is current
        TreeNode
        :return: tree size
        """
        ans = 1
        for c in self.iter_child():
            ans += c.size() if c is not None else 0
        return ans

    def depth(self):
        cur_max = 0
        for c in self.iter_child():
            c_depth = 0 if c is None else c.depth()
            cur_max = max(cur_max, c_depth)
        return cur_max + 1

    def to_tokens(self):
        if self._val is None or self._type is None:
            v = ""  # for debug"<none>"
        else:
            v = self._val
        if self._type == VAR:
            res = [v]
        elif self._type == OP:
            if len(self._children) == 1:
                res = [v]
                if self._children[0] is None:
                    print("WARNING: child is None")
                    res.append('null')
                else:
                    res.extend(self._children[0].to_tokens())
            else:
                c0 = self._children[0]
                res = [] if c0 is None else c0.to_tokens()
                res.append(v)
                if self._children[1] is None:
                    res.append('null')
                    print("WARNING: second child is None")
                else:
                    res.extend(self._children[1].to_tokens())
            if self._parent is not None and self._parent.get_type() != FUNC:
                res.insert(0, "(")
                res.append(")")
        else:
            # self._type == _FUNC:
            res = [v, "("]
            for i in self._children[:-1]:
                if i is None:
                    print("WARNING: child is None")
                    res.extend(['null', ','])
                else:
                    res.extend(i.to_tokens())
                    res.append(",")
            if self._children[-1] is not None:
                res.extend(self._children[-1].to_tokens())
            else:
                res.append("null")
            res.append(")")
        return res

    def get_postorder_list(self):
        order = list()
        for c in self._children:
            if c is not None:
                order.extend(c.get_postorder_list())
        order.append(self)
        return order

    def visualize(self, out_name=None, weights=None, vocab=None):
        # weights are post-order traversed
        if not GUI:
            return None
        if out_name is None:
            out_name = "graph"
        if weights is None:
            colors = np.zeros(self.size(), dtype=np.int)
        else:
            if not isinstance(weights, np.ndarray):
                weights = weights.clone().detach().numpy()
                weights = weights.reshape([-1])
            colors = np.digitize(weights, color_bin) - 1

        def fill_and_font(node_id):
            fill_color = color_map[colors[node_id]]
            if fill_color != color_map[-1]:
                font_color = 'black'
            else:
                font_color = 'white'
            return fill_color, font_color

        graph = Graph(filename=out_name, format='svg')
        if isinstance(self._val, int):
            val = vocab.id2word[self._val]
        else:
            val = self._val

        fill, font = fill_and_font(0)
        graph.node('0', val, style='filled', color=fill, fontcolor=font)
        node_id = 1
        stack = [(0, self)]
        while len(stack):
            par_id, par_node = stack.pop()
            for child in par_node.iter_child():
                if child is None:
                    continue
                val = child.get_value()
                if isinstance(val, int):
                    val = vocab.id2word[val]
                fill, font = fill_and_font(node_id)
                graph.node(str(node_id), val, style='filled', color=fill, fontcolor=font)
                graph.edge(str(par_id), str(node_id))
                stack.append((node_id, child))
                node_id += 1
        graph.render(out_name, view=False)

    def clone(self):
        new_root = TreeNode(val=self._val, t=self._type)
        for child in self._children:
            if child is None:
                new_root.add_child(None)
            else:
                new_root.add_child(child.clone())
        return new_root

    def reduce(self, vocab):
        if self._type == VAR:
            return
        reducible = True
        for c in self._children:
            if c is not None:
                c.reduce(vocab)
                if not vocab.is_const(c.get_value()):
                    reducible = False
                    # print(f"{c.get_value()} is not const, causing reduction stop")
        # print(self, reducible)
        if reducible:
            val = self.execute(vocab, {})
            self._type = VAR
            self._val = str(val)
            for c in self._children:
                if c is not None:
                    c.free_parent()
            self._children = list()
            # print("reduced to ", val)

        if self._val in ["<=", "<", ">=", ">", "==", "!="]:
            c1 = vocab.to_const(self._children[0].get_value())
            c2 = vocab.to_const(self._children[1].get_value())
            if c1 is not None:
                if self._children[1].get_value() == "+":
                    c21 = vocab.to_const(self._children[1].get_child(0).get_value())
                    c22 = vocab.to_const(self._children[1].get_child(1).get_value())
                    if c21 is not None:
                        self._children[0].set_value(str(c1 - c21))
                        sub_tree = self._children[1].get_child(1)
                    elif c22 is not None:
                        self._children[0].set_value(str(c1 - c22))
                        sub_tree = self._children[1].get_child(0)
                    else:
                        sub_tree = self._children[1]
                    sub_tree.free_parent()
                    self.set_child(1, sub_tree)
                elif self._children[1].get_value() == "-":
                    c21 = vocab.to_const(self._children[1].get_child(0).get_value())
                    c22 = vocab.to_const(self._children[1].get_child(1).get_value())
                    if c21 is not None:
                        sub_tree = self._children[1].get_child(1)
                        sub_tree.free_parent()
                        self._children[1].free_parent()
                        self.set_child(1, TreeNode(val=str(c21 - c1), t=VAR))

                        self._children[0].free_parent()
                        self.set_child(0, sub_tree)
                    elif c22 is not None:
                        self._children[0].set_value(str(c1 + c22))
                        sub_tree = self._children[1].get_child(0)
                        sub_tree.free_parent()
                        self.set_child(1, sub_tree)
            elif c2 is not None:
                if self._children[0].get_value() == "+":
                    c11 = vocab.to_const(self._children[0].get_child(0).get_value())
                    c12 = vocab.to_const(self._children[0].get_child(1).get_value())
                    if c11 is not None:
                        self._children[1].set_value(str(c2 - c11))
                        sub_tree = self._children[0].get_child(1)
                    elif c12 is not None:
                        self._children[1].set_value(str(c2 - c12))
                        sub_tree = self._children[0].get_child(0)
                    else:
                        sub_tree = self._children[0]
                    sub_tree.free_parent()
                    self.set_child(0, sub_tree)
                elif self._children[0].get_value() == "-":
                    c11 = vocab.to_const(self._children[0].get_child(0).get_value())
                    c12 = vocab.to_const(self._children[0].get_child(1).get_value())
                    if c11 is not None:
                        sub_tree = self._children[0].get_child(1)
                        sub_tree.free_parent()
                        self._children[0].free_parent()
                        self._children[0] = TreeNode(val=str(c11 - c2), t=VAR)
                        self._children[1].free_parent()
                        self.set_child(1, sub_tree)
                    elif c12 is not None:
                        self._children[1].set_value(str(c2 + c12))
                        sub_tree = self._children[0].get_child(0)
                        sub_tree.free_parent()
                        self.set_child(0, sub_tree)
            elif self._children[1].get_value() in "+-" and self._children[0].get_value() in "+-":
                c11 = vocab.to_const(self._children[0].get_child(0).get_value())
                c12 = vocab.to_const(self._children[0].get_child(1).get_value())
                c21 = vocab.to_const(self._children[1].get_child(0).get_value())
                c22 = vocab.to_const(self._children[1].get_child(1).get_value())
                if c11 is None and c12 is None:
                    return
                if c21 is None and c22 is None:
                    return
                if c11 is not None:
                    if c21 is not None:
                        if self._children[1].get_value() == "+":
                            self._children[0].get_child(0).set_value(str(c11 - c21))
                            sub_tree = self._children[1].get_child(1)
                            self._children[1].free_parent()
                            sub_tree.free_parent()
                            self.set_child(1, sub_tree)
                        elif self._children[1].get_value() == "-":
                            self._children[1].get_child(0).set_value(str(c21 - c11))
                            t22 = self._children[1].get_child(1)
                            t22.free_parent()
                            t11 = self._children[0].get_child(0)
                            t11.free_parent()
                            self._children[0].set_child(0, t22)
                            t21 = self._children[1].get_child(0)
                            t21.free_parent()
                            self.set_child(1, t21)
                    else:
                        if self._children[1].get_value() == "+":
                            self._children[0].get_child(0).set_value(str(c11 - c22))
                        elif self._children[1].get_value() == "-":
                            self._children[0].get_child(0).set_value(str(c11 + c22))
                        sub_tree = self._children[1].get_child(0)
                        self._children[1].free_parent()
                        sub_tree.free_parent()
                        self.set_child(1, sub_tree)
                elif c12 is not None:
                    if c21 is not None:
                        if self._children[0].get_value() == "-":
                            sub_tree = self._children[0].get_child(0)
                            sub_tree.free_parent()
                            self._children[0].free_parent()
                            self.set_child(0, sub_tree)
                            self._children[1].get_child(0).set_value(str(c21 + c12))
                        elif self._children[0].get_value() == "+":
                            sub_tree = self._children[0].get_child(0)
                            sub_tree.free_parent()
                            self._children[0].free_parent()
                            self.set_child(0, sub_tree)
                            self._children[1].get_child(0).set_value(str(c21 - c12))
                    elif c22 is not None:
                        if self._children[0].get_value() == "-":
                            sub_tree = self._children[0].get_child(0)
                            sub_tree.free_parent()
                            self._children[0].free_parent()
                            self.set_child(0, sub_tree)
                            if self._children[1].get_value() == "+":
                                self._children[1].get_child(1).set_value(str(c22 + c12))
                            else:
                                self._children[1].get_child(1).set_value(str(c22 - c12))
                        elif self._children[0].get_value() == "+":
                            if self._children[1].get_value() == "+":
                                sub_tree = self._children[0].get_child(0)
                                sub_tree.free_parent()
                                self._children[0].free_parent()
                                self.set_child(0, sub_tree)
                                self._children[1].get_child(1).set_value(str(c22 - c12))
                            elif self._children[1].get_value() == "-":
                                sub_tree = self._children[0].get_child(0)
                                sub_tree.free_parent()
                                self._children[0].free_parent()
                                self.set_child(0, sub_tree)
                                self._children[1].get_child(1).set_value(str(c22 + c12))


def token_tree_to_id_tree(root, vocab):
    """
    replace every val attributes in the tree with its id in vocab
    inplace operation, no new object is returned
    :param root: root of the tree
    :param vocab: vocab
    :return: None
    """
    if root is None:
        return
    if type(root.get_value()) != int:
        root.set_value(vocab.w2id(root.get_value()))
    for c in root.iter_child():
        token_tree_to_id_tree(c, vocab)


def id_tree_to_token_tree(root, vocab):
    """reverse method of token_tree_to_id_tree"""
    if root is None:
        return
    if type(root.get_value()) == int:
        root.set_value(vocab.id2w(root.get_value()))
    for c in root.iter_child():
        id_tree_to_token_tree(c, vocab)


def validate_tree(root, vocab):
    if root is None:
        return False
    if root.get_type() == VAR:
        return True
    if root.get_child_num() != vocab.num_operands(root.get_value()):
        return False
    if root.get_value() == "-":
        if not validate_tree(root.get_child(1), vocab):
            return False
        if root.get_child(0) is None:
            return True  # tested right child in last step
        return validate_tree(root.get_child(0), vocab)
    for leaf in root.iter_child():
        if not validate_tree(leaf, vocab):
            return False
    return True


def parse_expression(tokens, vocab):
    # print("parsing", tokens)
    flag = 0
    if len(tokens) == 0:
        return None
    if len(tokens) == 1:
        if vocab.is_const_or_var(tokens[0]):
            return TreeNode(val=tokens[0], t=VAR)
        else:
            raise ValueError(f"Invalid expression: {tokens} should be a const or var")
    if vocab.is_func(tokens[0]):
        root = TreeNode(val=tokens[0], t=FUNC)
        last = 2
        i = 1
        for i in range(1, len(tokens)):
            if tokens[i] == "(":
                flag += 1
            elif tokens[i] == ")":
                flag -= 1
                if flag == 0:
                    arg = parse_expression(tokens[last:i], vocab)
                    root.add_child(arg)
                    break
            elif tokens[i] == ",":
                if flag == 1:
                    arg = parse_expression(tokens[last:i], vocab)
                    root.add_child(arg)
                    last = i + 1
        if i == len(tokens) - 1:
            # pure function call structure
            return root
        else:
            # func() \op \arg structure
            if vocab.is_op(tokens[i + 1]) and not vocab.is_func(tokens[i + 1]):
                binary_root = TreeNode(tokens[i + 1])
                binary_root.set_type(OP)
                binary_root.set_value(tokens[i + 1])
                binary_root.add_child(root)
                if i + 2 == len(tokens):
                    raise ValueError("Empty Second args:" + "".join(tokens))
                if vocab.is_func(tokens[i + 2]) or len(tokens[i + 2:]) == 1:
                    arg = parse_expression(tokens[i + 2:], vocab)
                else:
                    if tokens[i + 2] != "(" or tokens[-1] != ")":
                        raise ValueError("Missing Parentheses:", tokens[i + 2:])
                    arg = parse_expression(tokens[i + 3: -1], vocab)
                binary_root.add_child(arg)
                return binary_root
    else:
        root = TreeNode(t=OP)
        # unary
        if vocab.is_op(tokens[0]) and not vocab.is_func(tokens[0]):
            if vocab.num_operands(tokens[0]) != 1 and tokens[0] != "-":
                raise ValueError(f"Invalid Expression: {tokens} should start with unary")
            root.set_value(tokens[0])
            if vocab.is_func(tokens[1]) or len(tokens[1:]) == 1:
                arg = parse_expression(tokens[1:], vocab)
            else:
                if tokens[1] != "(" or tokens[-1] != ")":
                    raise ValueError("Missing Parentheses:", tokens[1:])
                arg = parse_expression(tokens[2:-1], vocab)
            if tokens[0] == "-":
                root.add_child(None)
            root.add_child(arg)
            return root
        # binary
        for i in range(0, len(tokens)):
            if tokens[i] == "(":
                flag += 1
            elif tokens[i] == ")":
                flag -= 1
            elif vocab.is_op(tokens[i]) and not vocab.is_func(tokens[i]):
                if flag == 0:
                    root.set_value(tokens[i])
                    if tokens[0] == "(" and tokens[i - 1] == ")":
                        arg1 = parse_expression(tokens[1: i - 1], vocab)
                    else:
                        arg1 = parse_expression(tokens[: i], vocab)
                    root.add_child(arg1)
                    if i + 1 == len(tokens):
                        raise ValueError("Empty Second args:", tokens)
                    if vocab.is_func(tokens[i + 1]) or len(tokens[i + 1:]) == 1:
                        arg2 = parse_expression(tokens[i + 1:], vocab)
                    else:
                        if tokens[i + 1] != "(" or tokens[-1] != ")":
                            raise ValueError("Missing parentheses:", tokens)
                        arg2 = parse_expression(tokens[i + 2:-1], vocab)
                    root.add_child(arg2)
                    return root
        try:
            return parse_expression(tokens[1:-1], vocab)
        except ValueError:
            raise ValueError("Failed to locate root expression: " + " ".join(tokens))

