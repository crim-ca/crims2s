import torch.nn as nn


class ConvPostProcessing(nn.Module):
    def __init__(self, t2m_model, tp_model):
        super().__init__()

        self.t2m_model = t2m_model
        self.tp_model = tp_model

    def forward(self, example):
        t2m_dist = self.t2m_model(example)
        tp_dist = self.tp_model(example)

        return t2m_dist, tp_dist
