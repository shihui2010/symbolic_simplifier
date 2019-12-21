from typing import List
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class RewriteDecider(nn.Module):
    def __init__(self, hps, device=torch.device("cpu")):
        super(RewriteDecider, self).__init__()
        self._input_size = hps["encoder"]["output_size"]
        self.layers = nn.Sequential(
            nn.Linear(self._input_size, hps["rewriter_hidden_size"]),
            nn.Sigmoid(),
            nn.Linear(hps["rewriter_hidden_size"], 1),
            nn.Sigmoid()
        )
        self.to(device)
        self.optim = optim.Adam(self.parameters(), lr=hps["learning_rate"])

    def forward(self, rnn_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        :param rnn_outputs: list of [1, complexity_feature_hidden_state]
        :return:
        """
        outputs = list()
        for encoding in rnn_outputs:
            outputs.append(self.layers(encoding))
        outputs = torch.cat(outputs, dim=1)
        return outputs


if __name__ == "__main__":
    import numpy as np

    hps = dict(complexity_split=10, encoder=dict(output_size=36), rewriter_hidden_size=32)
    rnn_output = [torch.tensor(np.random.uniform(-50, 50, [1, 1, 36]), dtype=torch.float)
                  for _ in range(10)]
    classifier = RewriteDecider(hps)
    print(classifier(rnn_output).shape)
