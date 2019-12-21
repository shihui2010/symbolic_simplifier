from collections import MutableMapping
from Data.data_utils import TreeNode
from Optimizer.data_norm import *
# import z3


class Rule(object):
    def __init__(self, lhs: TreeNode, rhs: TreeNode, vocab=None, norm=True) -> None:
        """
        Rule Object, lhs is the original form of an expression, rhs is its simpler equivalence
        Rule is normalized.
        """
        assert lhs is not None, "Rule ValueError: empty lhs"
        assert rhs is not None, "Rule ValueError: empty rhs"

        self.lhs = lhs.clone()
        self.rhs = rhs.clone()

        if norm:
            assert vocab is not None, "must provide vocab for normalization"
            _, LUT = normalize(self.lhs, vocab)
            rename(self.rhs, LUT)

        self.lhs_str = str(self.lhs)
        self.rhs_str = str(self.rhs)
        self.__vocab = vocab

    def apply(self, *, expression=None, check_match=False) -> TreeNode:
        """
        assume the patterns of exp and self._lhs match, apply the transformation rule and returns new object
        :param expression: if expression is None, assume the pattern is pre-normalized and matches
        """
        if expression is None:
            assert not check_match, "expression not provided for check"
            return self.rhs.clone()

        _, LUT = normalize(expression, self.__vocab)
        if check_match:
            match = str(expression) == self.lhs_str
            if not match:
                res = expression
            else:
                res = self.rhs.clone()
        else:
            res = self.rhs.clone()
        denormalize(expression, LUT)
        denormalize(res, LUT)
        return res

    def matched(self, expression: TreeNode):
        """The whole sequence is the same with lhs after normalization"""

        _, LUT = normalize(expression, self.__vocab)
        res = str(expression) == self.lhs_str
        denormalize(expression, LUT)
        return res

    def soft_matched(self, exp: TreeNode):
        """
        The whole sequence match the lhs pattern with replacing subtrees as nodes
        assume common sub-expression folding is done
        :return:
        """
        exp = exp.clone()
        s_queue = [self.lhs]
        d_queue = [exp]
        LUT = dict()  # {str(var in lhs): origin tree in exp}
        rLUT = dict()  # {str(origin tree in exp): str(var in lhs)}
        while len(s_queue):
            node_s = s_queue.pop(0)
            node_d = d_queue.pop(0)
            if node_s is None and node_d is None:
                continue
            elif node_s is None or node_d is None:
                return False, None, None
            if node_s.get_child_num() == 0:
                # leaf node in the pattern
                if node_d.get_child_num() == 0:
                    if node_d.get_value() in rLUT:
                        if rLUT[node_d.get_value()] != node_s.get_value():
                            # renaming conflict
                            return False, None, None
                    if node_s.get_value() in LUT:
                        if str(LUT[node_s.get_value()]) != str(node_d):
                            # renaming conflict
                            return False, None, None
                    LUT[str(node_s)] = node_d.clone()
                    rLUT[str(node_d)] = str(node_s)
                    node_d.set_value(node_d.get_value())
                else:
                    # rename whole sub-tree in the exp
                    # rep can not in rLUT as common sub-expression folding is done
                    if str(node_s) in LUT:
                        # must be a rename conflict
                        return False, None, None
                    LUT[str(node_s)] = node_d.clone()
                    rLUT[str(node_d)] = str(node_s)
                    replace(node_d, node_s.clone())
            else:
                # function or operation
                if node_d.get_value() != node_s.get_value():
                    # operator must be exact same
                    return False, None, None
                for c in node_s.iter_child():
                    s_queue.append(c)
                for c in node_d.iter_child():
                    d_queue.append(c)
            if len(s_queue) != len(d_queue):
                return False, None, None
        return True, exp, LUT

    def soft_transform(self, exp: TreeNode):
        """ Assuming partially match is checked, so we only check soft match """
        matched, _, LUT = self.soft_matched(exp)
        if matched:
            return self._soft_rewrite(LUT)
        for cid, c in enumerate(exp.iter_child()):
            if c is None:
                continue
            new_c = self.soft_transform(c)
            exp.set_child(idx=cid, node=new_c)
        return exp

    def _soft_rewrite(self, LUT: dict):
        res = self.rhs.clone()
        if res.size() == 1:
            if res.get_value() in LUT:
                return LUT[res.get_value()]
            return res
        queue = [res]
        while len(queue):
            node = queue.pop(0)
            if node.size() == 1:
                replace(node, LUT[str(node)].clone())
            else:
                for c in node.iter_child():
                    if c is not None:
                        queue.append(c)
        return res

    def validate(self):
        pass

    def partially_match(self, expression: TreeNode):
        if self.matched(expression=expression):
            return True
        for c in expression.iter_child():
            if c is None:
                continue
            else:
                if self.partially_match(c):
                    return True
        return False

    def partially_transform(self, expression: TreeNode):
        expression = expression.clone()
        if self.matched(expression=expression):
            return self.apply(expression=expression, check_match=False)
        for cid, c in enumerate(expression.iter_child()):
            if c is None:
                continue
            else:
                new_c = self.partially_transform(c)
                expression.set_child(idx=cid, node=new_c)
        return expression

    def __hash__(self):
        return hash(self.lhs_str)

    def __repr__(self):
        return self.lhs_str + " -> " + self.rhs_str

    def dump(self):
        return [self.lhs.to_tokens(), self.rhs.to_tokens()]


class RuleSet(MutableMapping):
    def __init__(self, vocab, *args, **kwargs):
        self.__store = dict()    # {str: Rule}
        self.update(dict(*args, **kwargs))   # not actually used
        self.__vocab = vocab

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.__store[key]
        if not isinstance(key, TreeNode):
            raise TypeError("Not a valid type:", type(key))
        _, norm_key = self.__key_transform__(key)
        return self.__store[norm_key]

    def __key_transform__(self, key: TreeNode):
        tree_copy = key.clone()
        _, LUT = normalize(tree_copy, self.__vocab)
        return LUT, str(tree_copy)

    def __setitem__(self, key: TreeNode, value: TreeNode):
        """ set up a Transformation rule """
        rule = Rule(lhs=key, rhs=value, vocab=self.__vocab, norm=True)
        self.__store[rule.lhs_str] = rule

    def transform(self, tree: TreeNode) -> TreeNode:
        LUT, key = self.__key_transform__(tree)
        res = self.__store[key].apply()
        denormalize(res, LUT)
        return res

    def __iter__(self):
        return iter(self.__store)

    def __len__(self):
        return len(self.__store)

    def __delitem__(self, key: TreeNode):
        self.__store.pop(self.__key_transform__(key))

    def __dict__(self):
        return self.__store

    def __str__(self):
        return str(self.__store)

    def partially_match_rule(self, expression: TreeNode):
        """
        if expression partially matching some existing rule, return the key to get that rule
        :param expression:
        :return:
        """
        LUT, key = self.__key_transform__(expression)
        if key in self.__store:
            return self.__store[key]
        for c in expression.iter_child():
            if c is not None:
                res = self.partially_match_rule(c)
                if res is not None:
                    return res
        return None

    def soft_rewrite(self, exp: TreeNode):
        """
        if expression is "softly" matched to some rule, return transformed expression
        """
        str_ref = str(exp)
        exp_clone = exp.clone()
        for key in self.__store:
            exp_rewrite = self.__store[key].soft_transform(exp_clone)
            if str(exp_rewrite) != str_ref:
                return exp_rewrite
        return exp_clone

    def merge(self, rule_set):
        for k in rule_set:
            self.__store[k] = rule_set[k]

    def dump(self):
        rules = list()
        for rule in self.__store.values():
            rules.append(rule.dump())
        return rules
