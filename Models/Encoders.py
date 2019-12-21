import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, hps, device=torch.device("cpu")):
        super(EncoderLSTM, self).__init__()
        self._hps = hps
        self._device = device
        self.__build_graph()
        self.to(device)

    def __build_graph(self):
        """
        RNN encoder structure with embed layer,
        n-LSTM layer and projection layer
        """
        self.embed_matrix = nn.Parameter(
            torch.randn([self._hps["input_size"], self._hps["embed_size"]]))

        self.encoder = nn.LSTM(
            input_size=self._hps["embed_size"],
            hidden_size=self._hps["hidden_size"],
            num_layers=self._hps["num_layers"])

        self.projection = nn.Linear(
            self._hps["hidden_size"],
            self._hps["output_size"])

    def forward(self, input_seq):
        """
        :param input_seq: [seq_length, batch_size], indices of token
        :return: encoder outputs of seq_len, content vector
        """
        if type(input_seq) != torch.Tensor:
            input_seq = torch.from_numpy(input_seq)
        input_seq = input_seq.to(self._device)
        batch_first = torch.transpose(input_seq, 1, 0)
        embedded = F.embedding(batch_first, self.embed_matrix)
        embedded = torch.transpose(embedded, 1, 0)
        rnn_outputs, (last_h, last_c) = self.encoder(embedded)
        encoder_outputs = self.projection(rnn_outputs)
        return encoder_outputs, rnn_outputs, (last_h, last_c)


class NaryTreeLSTM(nn.Module):
    """N-ary Tree LSTM """
    def __init__(self, hps, device):
        super(NaryTreeLSTM, self).__init__()
        self._hps = hps
        self._device = device
        self.__build_graph()
        self.to(device)

    def __build_graph(self):
        self.embed_matrix = nn.Parameter(
            torch.randn([self._hps["input_size"], self._hps["embed_size"]]))

        self.wi = nn.Linear(self._hps["embed_size"], self._hps["hidden_size"])
        self.wf = nn.Linear(self._hps["embed_size"], self._hps["hidden_size"])
        self.wo = nn.Linear(self._hps["embed_size"], self._hps["hidden_size"])
        self.wu = nn.Linear(self._hps["embed_size"], self._hps["hidden_size"])

        self.bi = nn.Parameter(torch.randn([self._hps["hidden_size"]]))
        self.bf = nn.Parameter(torch.randn([self._hps["hidden_size"]]))
        self.bo = nn.Parameter(torch.randn([self._hps["hidden_size"]]))
        self.bu = nn.Parameter(torch.randn([self._hps["hidden_size"]]))

        self.ui = nn.ModuleList(
            [nn.Linear(self._hps["hidden_size"], self._hps["hidden_size"])
             for _ in range(self._hps["N"])])

        self.uf = nn.ModuleList(
            [nn.ModuleList(
                [nn.Linear(self._hps["hidden_size"], self._hps["hidden_size"])
                 for _ in range(self._hps["N"])])
             for _ in range(self._hps["N"])])

        self.uo = nn.ModuleList(
            [nn.Linear(self._hps["hidden_size"], self._hps["hidden_size"])
             for _ in range(self._hps["N"])])

        self.uu = nn.ModuleList(
            [nn.Linear(self._hps["hidden_size"], self._hps["hidden_size"])
             for _ in range(self._hps["N"])])

        self.empty_hc = torch.zeros([1, self._hps["hidden_size"]]).to(self._device)
        self.empty_out = torch.zeros([1, self._hps["output_size"]]).to(self._device)

        self.projection = nn.Linear(
            self._hps["hidden_size"], self._hps["output_size"])

    def _node_forward(self, h_children, c_children, xj):
        """
        A single node, not a batch of
        :param h_children: list of N h_l of child nodes
        :param c_children: list of N c_l of child nodes
        :param xj: embedded input of current node
        :return: hj, cj
        """
        sum_i, sum_o, sum_u = list(), list(), list()
        sum_f = list()
        for l in range(self._hps["N"]):
            sum_i.append(self.ui[l](h_children[l]))
            sum_o.append(self.uo[l](h_children[l]))
            sum_u.append(self.uu[l](h_children[l]))
            sum_fl = list()
            for l_prime in range(self._hps["N"]):
                sum_fl.append(self.uf[l][l_prime](h_children[l_prime]))
            sum_f.append(torch.sum(torch.stack(sum_fl, dim=0), dim=0))

        # not activated yet
        sum_i = torch.stack(sum_i, dim=0)
        sum_o = torch.stack(sum_o, dim=0)
        sum_u = torch.stack(sum_u, dim=0)
        ij = self.wi(xj) + torch.sum(sum_i, dim=0) + self.bi
        oj = self.wo(xj) + torch.sum(sum_o, dim=0) + self.bo
        uj = self.wu(xj) + torch.sum(sum_u, dim=0) + self.bu
        fj = list()

        for k in range(self._hps["N"]):
            fj.append(torch.sigmoid(self.wf(xj) + sum_f[k] + self.bf))

        # activate
        ij = torch.sigmoid(ij)
        oj = torch.sigmoid(oj)
        uj = torch.tanh(uj)

        fc = list()
        for k in range(self._hps["N"]):
            fc.append(fj[k] * c_children[k])
        cj = ij * uj + torch.sum(torch.stack(fc, dim=0), dim=0)
        hj = oj * torch.tanh(cj)
        return cj, hj

    def forward(self, root):
        """
        not a batched method
        :param root: TreeNode object
        :return: outputs of each node, (h, c)
        """
        encoder_outputs = list()
        rnn_outputs = list()
        input_ref = list()
        h_children, c_children = list(), list()
        idx = 0

        # DFS
        for child_node in root.iter_child():
            if child_node is None:
                h_children.append(self.empty_hc)
                c_children.append(self.empty_hc)
                continue
            o_l, hs_l, in_l, (h_l, c_l) = self.forward(child_node)

            encoder_outputs.extend(o_l)
            rnn_outputs.extend(hs_l)
            input_ref.extend(in_l)

            h_children.append(h_l)
            c_children.append(c_l)
            idx += 1

        while idx < self._hps["N"]:
            h_children.append(self.empty_hc)
            c_children.append(self.empty_hc)
            idx += 1

        # print root.get_value()
        input_j = torch.tensor([root.get_value()]).to(self._device)
        xj = F.embedding(input_j, self.embed_matrix)
        cj, hj = self._node_forward(h_children, c_children, xj)
        out_j = self.projection(hj)

        encoder_outputs.append(out_j)
        rnn_outputs.append(hj)
        input_ref.append(root.get_value())
        # return out_j, (hj, cj)
        return encoder_outputs, rnn_outputs, input_ref, (hj, cj)
    # return encoder_outputs, rnn_outputs, (last_h, last_c)
