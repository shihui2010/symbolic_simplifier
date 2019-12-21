from Data.data_utils import Vocab


class HalideVocab(Vocab):
    _ops = ["min", "max", ">=", "<=", "<", ">", "==", "!", "!=", "select",
            "+", "-", "*", "/", "&&", "||"]
    #  % disabled
    _probs = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
              0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    _funcs = ["min", "max", "select"]

    def __init__(self):
        super(HalideVocab, self).__init__()
        base = len(self.word2id)
        self.word2id.update({f: i + base for i, f in enumerate(self._ops)})
        self._auto_id2word()
        self.op_num = len(self._ops)

    def num_operands(self, token):
        if type(token) == int:
            if token not in self.id2word:
                raise ValueError(str(token) + "is not a known token")
            token = self.id2word[token]
        if token not in self._ops:
            return 0
        if token == "select":
            return 3
        if token == "!":
            return 1
        return 2

    def get_ops(self):
        # copy
        return [i for i in self._ops]

    def is_op(self, pattern):
        if type(pattern) == int:
            if pattern in self.id2word:
                return self.id2word[pattern] in self._ops
            return False
        return pattern in self._ops

    def is_func(self, pattern):
        if type(pattern) == int:
            if pattern in self.id2word:
                return self.id2word[pattern] in self._funcs
            return False
        return pattern in self._funcs

    def execute(self, op, *args):
        # print args
        if op not in self._ops:
            raise ValueError("Invalid operation:", op)
        if op in ["!", "&&", "||"]:
            args = [bool(i) for i in args]
        elif op == "select":
            new_args = [bool(args[0])]
            for i in args[1:]:
                if isinstance(i, bool):
                    new_args.append(int(i))
                else:
                    new_args.append(i)
            args = new_args
        else:
            new_args = list()
            for i in args:
                if isinstance(i, bool):
                    new_args.append(int(i))
                else:
                    new_args.append(i)
            args = new_args
        # print(op, args)
        if op == "min":
            return min(args[0], args[1])
        if op == "max":
            return max(args[0], args[1])
        if op == ">=":
            return args[0] >= args[1]
        if op == "<=":
            return args[0] <= args[1]
        if op == "<":
            return args[0] < args[1]
        if op == ">":
            return args[0] > args[1]
        if op == "==":
            return args[0] == args[1]
        if op == "!":
            return not args[0]
        if op == "!=":
            return args[0] != args[1]
        if op == "select":
            return args[1] if args[0] else args[2]
        if op == "+":
            return args[0] + args[1]
        if op == "-":
            if len(args) == 1:
                return - args[0]
            else:
                return args[0] - args[1]
        if op == "*":
            return args[0] * args[1]
        if op == "/":
            if abs(args[1] - args[0]) < 1e-7:
                return 1.0
            # if args[1] == 0:
            #     return 0
            return args[0] / args[1]
        if op == "%":
            if args[1] == 0:
                return 0
            return args[0] % args[1]
        if op == "&&":
            return args[0] and args[1]
        if op == "||":
            return args[0] or args[1]
        raise ValueError(f"Op not implemented:{op}")


if __name__ == "__main__":
    # test execution
    print("Running Execution Test:")
    vocab = HalideVocab()
    ops = ["min", "max", ">=", "<=", "<", ">", "==", "!", "!=", "select",
           "+", "-", "*", "/", "%", "&&", "||"]
    ans = [4, 9, False, True, True, False, False, False, True, 9,
           13, -5, 36, 0, 4, True, True]
    arg1 = 4
    arg2 = 9
    arg3 = 3
    for op, res in zip(ops, ans):
        r = vocab.execute(op, arg1, arg2, arg3)
        assert r == res, "Execution error: op=%s, arg1=%d, arg2=%d, "\
            "arg3=%d, res=" % (op, arg1, arg2, arg3) + str(r)
    print(len(ops), "Ops tested, all passed")


