import torch
import torch.nn as nn
import torch.nn.functional as F


class InputAttention(nn.Module):
    """
    apply attention to get weighted context vector before fed into decoder
    input: embedded input, encoder states, prev_decoder_state
    output the attended input for decoder rnn
    """
    def __init__(self, hps, device):
        super(InputAttention, self).__init__()
        self._hps = hps
        self._device = device
        self.__build_graph()
        self.to(device)

    def __build_graph(self):
        self.attention = nn.Linear(
            self._hps["embed_size"] + self._hps["hidden_size"],
            self._hps["attention_len"])

        self.decoder_input_projection = nn.Linear(
            self._hps["embed_size"] + self._hps["encoder_out_size"],
            self._hps["embed_size"])

    def forward(self, hidden_state, embedded, encoder_outputs):
        """
        :param hidden_state: previous hidden states
        :param embedded: embedded input, batch size as second dimension
        :param encoder_outputs: encoder outputs for applying attention
        :return: attention combined
        """
        attention_input = torch.cat([embedded, hidden_state[0]], dim=-1)
        attention = self.attention(attention_input)
        attention_weights = F.softmax(attention, dim=-1)

        # attention weights: [1, batch_size, max_seq_len]
        # encoder outputs: [max_seq_len, batch_size, encoder_output_size]

        transposed_outputs = encoder_outputs
        # transposed_outputs: [batch_size, max_seq_len, encoder_output_size]

        transposed_weights = torch.transpose(
            attention_weights, dim0=0, dim1=1)
        # transposed_weights: [batch_size, 1, max_seq_len]

        weighted_states = torch.bmm(
            transposed_weights, transposed_outputs)
        # weighted_states: [batch_size, 1, encoder_output_size]

        states_sum = torch.transpose(
            weighted_states, dim0=0, dim1=1)
        projected_rnn_inputs = self.decoder_input_projection(
            torch.cat([embedded, states_sum], dim=-1))
        return attention_weights, projected_rnn_inputs


class ProjectionAttention(nn.Module):
    """
    apply attention to decoder outputs. weighted context is then fed into projection layer
    input: encoder hidden_states, decoder state,
    """
    def __init__(self, hps, device):
        super(ProjectionAttention, self).__init__()
        self._hps = hps
        self._device = device
        self.__build_graph()
        self.to(device)

    def __build_graph(self):
        # attention
        self.v_att = nn.Linear(self._hps["att_hidden_size"], 1, bias=False)

        self.wh_att = nn.Linear(self._hps["encoder_hidden_size"],
                                self._hps["att_hidden_size"], bias=False)

        self.ws_att = nn.Linear(self._hps["hidden_size"] * 2,
                                self._hps["att_hidden_size"], bias=False)

        self.b_att = nn.Parameter(torch.randn([self._hps["att_hidden_size"]]))
        self.output_size = self._hps["encoder_hidden_size"] + 2 * self._hps["hidden_size"]

    def forward(self, s_t, encoder_hidden_states):
        """
        perform attention weighed context for projection layer
        :param s_t: concated (h_s, c_s) of decode rnn
        :param encoder_hidden_states: encoder_hidden_states being input_seq long
        :return: weighted and projected context for final projection
        """
        # attention use only h in encoder outputs
        e_t = self.v_att(
            torch.tanh(
                self.wh_att(encoder_hidden_states) +
                self.ws_att(s_t) + self.b_att
            )
        )
        attention = F.softmax(e_t, dim=0)
        # attention is [seq_len, batch_size, 1]

        # p_vocab by projection
        ht_star = torch.sum(attention * encoder_hidden_states, dim=0)
        # ht_star = [batch_size, encoder_hidden_size]
        state_cat = torch.cat((ht_star, s_t), dim=1)
        # state_cat = [batch_size,
        # encoder_hidden_size + 2 * decoder_hidden_size]

        return attention, ht_star, state_cat


class Pointer(nn.Module):
    """
    pointer p_gen takes:
     weighted encoder states h_star,
     decoder state s_t
     encoder input x_t
    generate the p_gen
    """
    def __init__(self, hps, device):
        super(Pointer, self).__init__()
        self._hps = hps
        self._device = device
        self.__build_graph()
        self.to(device)

    def __build_graph(self):
        # p_gen
        self.wh_gen = nn.Linear(self._hps["encoder_hidden_size"], 1, bias=False)
        self.ws_gen = nn.Linear(self._hps["hidden_size"] * 2, 1, bias=False)
        self.wx_gen = nn.Linear(self._hps["embed_size"], 1, bias=False)
        self.b_gen = nn.Parameter(torch.randn([1]))

    def forward(self, ht_star, s_t, embedded):
        """
        :param ht_star: weighted encoder states
        :param s_t: decoder state (concat) at time t
        :param embedded: embedded input for decoder at time t
        :return: p_gen
        """

        # p_gen
        p_gen = torch.sigmoid(
            self.wh_gen(ht_star) + self.ws_gen(s_t) + self.wx_gen(embedded)
            + self.b_gen)
        return p_gen