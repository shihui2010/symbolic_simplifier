from Models.Attentions import *


class AttentionDecoderLSTM(nn.Module):
    def __init__(self, hps, device=torch.device("cpu"), embed_matrix=None):
        super(AttentionDecoderLSTM, self).__init__()
        self._hps = hps
        self._device = device
        self.__build_graph(embed_matrix=embed_matrix)
        self.to(device)

    def __build_graph(self, embed_matrix=None):
        """
        A decoder consist of:
            a embedding layer, different from the encoder's
            a decode Rnn, the initial state of which is the last hidden state
                of encoder
            a projection and softmax layer to predict logits
        :return:
        """
        if embed_matrix is not None:
            self.embed_matrix = nn.Parameter(
                torch.randn([self._hps["input_size"], self._hps["embed_size"]]))
        else:
            # shared embedding
            self.embed_matrix = embed_matrix

        self.attention = InputAttention(self._hps, self._device)

        self.decoder = nn.LSTM(
            input_size=self._hps["embed_size"],
            hidden_size=self._hps["hidden_size"],
            num_layers=self._hps["num_layers"])

        self.projection = nn.Linear(
            self._hps["hidden_size"],
            self._hps["output_size"])

    def forward(self, hidden_state, input_seq, encoder_outputs):
        """
        :param hidden_state:
            if t = 0, state is last hidden state of encoder
            if t != 0, state is prev hidden state of itself
            hidden_state = (last_h, last_c)
        :param input_seq: tokens, seq_len as first dimension
        :param encoder_outputs: outputs of encoder at each t, distrib prob
        :return:
        """
        if type(input_seq) != torch.Tensor:
            input_seq = torch.from_numpy(input_seq)
        input_seq = input_seq.type(torch.LongTensor)
        input_seq = input_seq.to(self._device)
        # batch_first = torch.transpose(input_seq, 1, 0)
        batch_first = input_seq
        # print(batch_first.shape)
        embedded = F.embedding(batch_first, self.embed_matrix)
        # print(embedded.shape, hidden_state[0].shape)
        # embedded = torch.transpose(embedded, 1, 0)
        # embedded : [seq=1, batch_size, embed_size]
        hs = (hidden_state[0][:, -1:, :], hidden_state[1][:, -1:, :])

        att_weights, projected_rnn_inputs = self.attention(hs, embedded, encoder_outputs)

        # print("projected rnn", projected_rnn_inputs.shape)
        rnn_outputs, (last_h, last_c) = self.decoder(
            projected_rnn_inputs, hs)

        decoder_output = self.projection(rnn_outputs)
        outputs = F.softmax(decoder_output, dim=-1)
        return att_weights, outputs, (last_h, last_c)


class PointerDecoder(nn.Module):
    def __init__(self, hps, device, embed_matrix=None):
        super(PointerDecoder, self).__init__()
        self._hps = hps
        self._device = device
        self.__build_graph(embed_matrix=embed_matrix)
        self.to(device)

    def __build_graph(self, embed_matrix=None):
        """
        n-LSTM layer decoder with attention and pointers
        using concatenation of c and h for s_t (decoder state)
        """
        if embed_matrix is None:
            self.embed_matrix = nn.Parameter(
                torch.randn([self._hps["input_size"], self._hps["embed_size"]]))
        else:
            # shared embedding
            self.embed_matrix = embed_matrix

        self.p_gen = Pointer(self._hps, self._device)

        # decoder rnn
        self.decoder = nn.LSTM(
            input_size=self._hps["embed_size"],
            hidden_size=self._hps["hidden_size"],
            num_layers=self._hps["num_layers"])

        self.attention = ProjectionAttention(self._hps, self._device)

        # projection
        self.projection1 = nn.Linear(
            self._hps["hidden_size"] * 2 + self._hps["encoder_hidden_size"],
            self._hps["proj_hidden_size"])  # with bias
        self.projection2 = nn.Linear(
            self._hps["proj_hidden_size"],
            self._hps["output_size"])  # with bias

    def forward(self, hidden_state, input_seq, encoder_hidden_states,
                encoder_extend_vocab):
        """
        :param hidden_state:
            if t = 0: last hidden state of encoder
            if t != 0: previous hidden state of decoder
        :param input_seq: tokens with 1 being first dimension (1-step decode)
        :param encoder_hidden_states: outputs of encoder at each step=i
            shape = [input_seq_len, batch_size, encoder_hidden_size]
        :param encoder_extend_vocab:
            input sequence vocab idx in extended vocab
            use as index for scatter_add
        :return: logits = p_gen * decoder_output + (1 - p_gen) * pointer
        """
        if type(input_seq) != torch.Tensor:
            input_seq = torch.from_numpy(input_seq)
        input_seq = input_seq.type(torch.LongTensor)
        input_seq = input_seq.to(self._device)
        embedded = F.embedding(input_seq, self.embed_matrix)

        # forward one step decoding
        outputs, (h_t, c_t) = self.decoder(embedded, hidden_state)

        s_t = torch.cat((h_t.view(-1, self._hps["hidden_size"]),
                         c_t.view(-1, self._hps["hidden_size"])), 1)

        attention, ht_star, state_cat = self.attention(s_t, encoder_hidden_states)

        p_vocab = F.softmax(
            self.projection2(
                torch.sigmoid(
                    self.projection1(state_cat))),
            dim=-1)
        # p_vocab = [batch_size, vocab_size]

        p_gen = self.p_gen(ht_star, s_t, embedded)

        # overall distribution
        vocab_dist = p_gen * p_vocab
        att_dist = (1 - p_gen) * attention
        extra_zeros = torch.zeros(1, vocab_dist.shape[1],  # batch_size
                                  self._hps["oov_size"]).to(self._device)
        vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=2)
        shp = encoder_extend_vocab.shape
        if type(encoder_extend_vocab) != torch.Tensor:
            encoder_extend_vocab = torch.from_numpy(encoder_extend_vocab)
        encoder_extend_vocab = encoder_extend_vocab.type(torch.LongTensor)
        encoder_extend_vocab = encoder_extend_vocab.to(self._device)
        encoder_extend_vocab = encoder_extend_vocab.view([shp[0], shp[1], 1])
        idx = torch.transpose(encoder_extend_vocab, 0, 2)
        att_trans = torch.transpose(att_dist, 0, 2)
        final_dist = vocab_dist.scatter_add(2, idx, att_trans)
        return attention, final_dist, (h_t, c_t)


class NaryTreeDecoder(nn.Module):
    def __init__(self, hps, device, embed_matrix=None):
        super(NaryTreeDecoder, self).__init__()
        self._hps = hps
        self._device = device
        self._set_embedding(embed_matrix=embed_matrix)
        self._build_graph()
        self._set_mask()
        self.to(device)

    def _set_mask(self):
        # set up mask, masking out <s>, <pad>, (, ), ","
        mask = [1] * self._hps["output_size"]
        for idx in [0, 1, 3, 4, 5]:
            mask[idx] = 0
        # output mask
        self.mask = torch.tensor(mask, dtype=torch.float).to(self._device)

    def _set_embedding(self, embed_matrix):
        if embed_matrix is None:
            self.embed_matrix = nn.Parameter(
                torch.randn([self._hps["input_size"], self._hps["embed_size"]]))
        else:
            # shared embedding
            self.embed_matrix = embed_matrix

    def _tree_rnn(self, embed_size, hidden_size):
        # input gate parameters, child idx dependent
        self.wi_list = nn.ModuleList(
            [nn.Linear(embed_size, hidden_size, bias=False)
             for _ in range(self._hps["N"] + 1)])
        self.ui_list = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=True)  # include b_ik
             for _ in range(self._hps["N"] + 1)])

        # forget gate parameters, child idx dependent
        self.wf_list = nn.ModuleList(
            [nn.Linear(embed_size, hidden_size, bias=False)
             for _ in range(self._hps["N"] + 1)])
        self.uf_list = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=True)
             for _ in range(self._hps["N"] + 1)])  # include b_fk

        # output gate parameters, child idx dependent
        self.wo_list = nn.ModuleList(
            [nn.Linear(embed_size, hidden_size, bias=False)
             for _ in range(self._hps["N"] + 1)])
        self.uo_list = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=True)
             for _ in range(self._hps["N"] + 1)])  # include b_ok

        # input u, child idx dependent
        self.wu_list = nn.ModuleList(
            [nn.Linear(embed_size, hidden_size, bias=False)
             for _ in range(self._hps["N"] + 1)])
        self.uu_list = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias=True)
             for _ in range(self._hps["N"] + 1)])  # include b_uk

        self.h_projs = nn.ModuleList(
            [nn.Linear(2 * hidden_size, hidden_size, bias=True)
             for _ in range(self._hps["N"])])
        self.c_projs = nn.ModuleList(
            [nn.Linear(2 * hidden_size, hidden_size, bias=True)
             for _ in range(self._hps["N"])])
        # for combining parent states with child states

    def _build_graph(self):
        self._tree_rnn(self._hps["embed_size"], self._hps["hidden_size"])

        # output projection
        self.projection = nn.Linear(self._hps["hidden_size"], self._hps["output_size"])

    def _forward_rnn(self, embedded, hidden_state):
        """
        process tree rnn
        """
        embedded = embedded.view([1, -1])
        # print("hidden_state", hidden_state[0].shape)
        # print("embedded", embedded.shape)
        c_list, h_list = list(), list()
        for k in range(self._hps["N"] + 1):
            i_jk = torch.sigmoid(
                self.wi_list[k](embedded) + self.ui_list[k](hidden_state[0]))
            f_jk = torch.sigmoid(
                self.wf_list[k](embedded) + self.uf_list[k](hidden_state[0]))
            o_jk = torch.sigmoid(
                self.wo_list[k](embedded) + self.uo_list[k](hidden_state[0]))
            u_tk = torch.tanh(
                self.wu_list[k](embedded) + self.uu_list[k](hidden_state[0]))
            c_list.append(
                i_jk * u_tk + f_jk * hidden_state[1])
            h_list.append(
                o_jk * torch.tanh(c_list[-1]))

        for i in range(self._hps["N"]):
            # print(h_list[0].shape, h_list[i + 1].shape)
            h_list[i + 1] = self.h_projs[i](
                torch.cat([h_list[0], h_list[i + 1]], dim=-1))
            c_list[i + 1] = self.c_projs[i](
                torch.cat([c_list[0], c_list[i + 1]], dim=-1))
        return h_list, c_list

    def forward(self, hidden_state, input_seq, *args, **kwargs):
        """
        :param hidden_state:
        :param input_seq: yt for LTSM input
        :return: probs on vocab, (h_t * N, c_t * N)
        """
        if type(input_seq) != torch.Tensor:
            input_seq = torch.from_numpy(input_seq)
        input_seq = input_seq.type(torch.LongTensor)
        # print "input_seq shape", input_seq.shape
        input_seq = input_seq.to(self._device)
        embedded = F.embedding(input_seq, self.embed_matrix)

        h_list, c_list = self._forward_rnn(embedded, hidden_state)

        # projection
        proj = self.projection(h_list[0])

        # mask before softmax
        output = proj + (self.mask + 1e-7).log()

        output = F.softmax(output, dim=-1)
        return output, (h_list[1:], c_list[1:])


class AttentionTreeDecoder(NaryTreeDecoder):
    def __init__(self, hps, device, embed_matrix=None):
        super(AttentionTreeDecoder, self).__init__(hps, device, embed_matrix=embed_matrix)

    def _build_graph(self):
        embed_size = self._hps["embed_size"]
        hidden_size = self._hps["hidden_size"]

        # rnn layer
        self._tree_rnn(embed_size, hidden_size)

        # attention
        self.attention = InputAttention(self._hps, self._device)

        # output projection
        self.projection = nn.Linear(hidden_size, self._hps["output_size"])

    def forward(self, hidden_state, input_seq, *args, **kwargs):
        """
        :param hidden_state:
        :param input_seq: yt for LTSM input
        :return: probs on vocab, (h_t * N, c_t * N)
        """
        if "encoder_outputs" not in kwargs:
            raise TypeError("Missing parameter encoder_outputs")
        encoder_outputs = kwargs["encoder_outputs"]

        if type(input_seq) != torch.Tensor:
            input_seq = torch.from_numpy(input_seq)
        input_seq = input_seq.type(torch.LongTensor)
        # print "input_seq shape", input_seq.shape
        input_seq = input_seq.to(self._device)
        embedded = F.embedding(input_seq, self.embed_matrix)

        hidden_state = (hidden_state[0].view(1, 1, -1), hidden_state[1].view(1, 1, -1))
        # print embedded.shape, hidden_state[0].shape, encoder_outputs.shape
        att, projected_rnn_inputs = self.attention(hidden_state, embedded, encoder_outputs)

        h_list, c_list = self._forward_rnn(projected_rnn_inputs, hidden_state)

        # projection
        proj = self.projection(h_list[0])

        # mask before softmax
        output = proj + (self.mask + 1e-45).log()

        output = F.softmax(output, dim=-1)

        return att, output, (h_list[1:], c_list[1:])


class ProjAttTreeDecoder(NaryTreeDecoder):
    def __init__(self, hps, device, embed_matrix=None):
        super(ProjAttTreeDecoder, self).__init__(hps, device, embed_matrix)

    def _build_graph(self):
        embed_size = self._hps["embed_size"]
        hidden_size = self._hps["hidden_size"]

        # rnn layer
        self._tree_rnn(embed_size, hidden_size)

        # attention
        self.attention = ProjectionAttention(self._hps, self._device)

        # output projection
        self.projection = nn.Linear(self.attention.output_size, self._hps["output_size"])

    def forward(self, hidden_state, input_seq, *args, **kwargs):
        """
        :param hidden_state:
        :param input_seq: yt for LTSM input
        :return: probs on vocab, (h_t * N, c_t * N)
        """
        if "encoder_hidden_states" not in kwargs:
            raise TypeError("Missing parameter encoder_hidden_states")
        encoder_hidden_states = kwargs["encoder_hidden_states"]

        if type(input_seq) != torch.Tensor:
            input_seq = torch.from_numpy(input_seq)
        input_seq = input_seq.type(torch.LongTensor)
        # print "input_seq shape", input_seq.shape
        input_seq = input_seq.to(self._device)
        embedded = F.embedding(input_seq, self.embed_matrix)

        h_list, c_list = self._forward_rnn(embedded, hidden_state)

        s_t = torch.cat([h_list[0].view(-1, self._hps["hidden_size"]),
                         c_list[0].view(-1, self._hps["hidden_size"])], 1)
        # print "s_t shape", s_t.shape

        attention, ht_star, state_cat = self.attention(s_t, encoder_hidden_states)
        # print "state_cat shape", state_cat.shape

        p_out = self.projection(state_cat) + (self.mask + 1e-45).log()
        p_vocab = F.softmax(p_out, dim=-1)
        p_vocab = p_vocab.unsqueeze(dim=1)
        return attention, p_vocab, (h_list[1:], c_list[1:])


class PointerTreeDecoder(NaryTreeDecoder):
    def __init__(self, hps, device, embed_matrix=None):
        super(PointerTreeDecoder, self).__init__(hps, device, embed_matrix)

    def _set_mask(self):
        """set up long mask: output size + oov size"""
        # set up mask, masking out <s>, <pad>, (, ), ","
        mask = [1] * (self._hps["output_size"] + self._hps["oov_size"])
        for idx in [0, 1, 3, 4, 5]:
            mask[idx] = 0
        # output mask
        self.mask = torch.tensor(mask, dtype=torch.float).to(self._device)
        self.short_mask = self.mask[:self._hps["output_size"]]

    def _build_graph(self):
        embed_size = self._hps["embed_size"]
        hidden_size = self._hps["hidden_size"]

        # rnn layer
        self._tree_rnn(embed_size, hidden_size)

        # attention
        self.attention = ProjectionAttention(self._hps, self._device)
        self.p_gen = Pointer(self._hps, self._device)

        # output projection
        self.projection = nn.Linear(self.attention.output_size, self._hps["output_size"])

    def forward(self, hidden_state, input_seq, *args, **kwargs):
        """
        :param hidden_state:
        :param input_seq: yt for LTSM input
        :return: probs on vocab, (h_t * N, c_t * N)
        """
        if "encoder_hidden_states" not in kwargs:
            raise TypeError("Missing parameter encoder_hidden_states")
        encoder_hidden_states = kwargs["encoder_hidden_states"]
        if "encoder_extend_vocab" not in kwargs:
            raise TypeError("Missing parameter encoder extend vocab")
        encoder_extend_vocab = kwargs["encoder_extend_vocab"]

        if type(input_seq) != torch.Tensor:
            input_seq = torch.from_numpy(input_seq)
        input_seq = input_seq.type(torch.LongTensor)
        # print "input_seq shape", input_seq.shape
        input_seq = input_seq.to(self._device)
        embedded = F.embedding(input_seq, self.embed_matrix)

        h_list, c_list = self._forward_rnn(embedded, hidden_state)

        s_t = torch.cat([h_list[0].view(-1, self._hps["hidden_size"]),
                         c_list[0].view(-1, self._hps["hidden_size"])], 1)
        # print "s_t shape", s_t.shape

        attention, ht_star, state_cat = self.attention(s_t, encoder_hidden_states)
        # print "state_cat shape", state_cat.shape

        # mask p_vocab
        p_out = self.projection(state_cat) + (self.short_mask + 1e-45).log()
        p_vocab = F.softmax(p_out, dim=-1)
        # p_vocab = [batch_size, vocab_size]

        p_gen = self.p_gen(ht_star, s_t, embedded)
        # overall distribution
        vocab_dist = p_gen * p_vocab

        extra_zeros = torch.zeros(1, vocab_dist.shape[1],  # batch_size
                                  self._hps["oov_size"]).to(self._device)
        vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=2)
        shp = encoder_extend_vocab.shape
        if type(encoder_extend_vocab) != torch.Tensor:
            encoder_extend_vocab = torch.from_numpy(encoder_extend_vocab)
        encoder_extend_vocab = encoder_extend_vocab.type(torch.LongTensor)
        encoder_extend_vocab = encoder_extend_vocab.to(self._device)
        encoder_extend_vocab = encoder_extend_vocab.view([shp[0], shp[1], 1])
        idx = torch.transpose(encoder_extend_vocab, 0, 2)
        att_trans = torch.transpose(attention, 0, 2)

        # initialize zero tensor for pointer distribution
        p_point = torch.zeros_like(vocab_dist)
        p_dist = p_point.scatter_add(2, idx, att_trans)
        # mask again
        p_dist = p_dist * self.mask
        # renormalize
        p_renorm = torch.sum(p_dist, dim=-1)
        # final_dist = vocab_dist.scatter_add(2, idx, att_trans)
        final_dist = vocab_dist + p_dist / p_renorm * (1 - p_gen)
        # print(final_dist, torch.sum(final_dist))
        # input()
        return attention, final_dist, (h_list[1:], c_list[1:])
