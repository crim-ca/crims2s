import torch
import torch.nn as nn

from .bayes import Projection
from .conv import DistributionConvPostProcessing, CommonTrunk
from .util import DistributionModelAdapter


class PretrainedModelWrapper(nn.Module):
    def __init__(
        self, pretrained_model, in_features, adapter_embedding=64, adapter_n_blocks=1
    ):
        super().__init__()

        self.proj1 = Projection(in_features, adapter_embedding, flatten_time=False)
        self.trunk = CommonTrunk(
            adapter_embedding, n_blocks=adapter_n_blocks, dropout=0.1
        )
        self.proj2 = Projection(adapter_embedding, 3, flatten_time=True, width=3)

        self.pretrained = pretrained_model

    def forward(self, x):
        x = torch.transpose(x, -1, 1)  # Swap channels and time dim.

        x = self.proj1(x)
        x = self.trunk(x)
        x = self.proj2(x)

        x = x.mean(-1)  # Remove time dimension (which was flattened by the projection).

        x = self.pretrained.forward(x)["out"]

        batch_size = x.shape[0]

        x_t2m = x[:, :4]
        x_tp = x[:, 4:]

        x_t2m = x_t2m.reshape(batch_size, 2, 121, 240, 2)
        x_tp = x_tp.reshape(batch_size, 2, 121, 240, 2)

        return x_t2m, x_tp


class FCNPreTrained(PretrainedModelWrapper):
    """FCN Pretrained model that outputs 8 channels. There are 4 channels to t2m and 
    4 channels for tp. For t2m there are 2 channels per lead time. For every lead time 
    there is one channel for adjusting mu and one channel for adjusting sigma."""

    def __init__(self, *args, **kwargs):
        fcn = torch.hub.load("pytorch/vision:v0.10.0", "fcn_resnet50", pretrained=True)
        fcn.classifier[4] = nn.Conv2d(512, 8, kernel_size=3, stride=1)

        super().__init__(fcn, *args, **kwargs)


class DistributionFCNPostProcessing(DistributionConvPostProcessing):
    """FCN Post processing model that outputs distributions."""

    def __init__(self, *args, **kwargs):
        fcn = FCNPreTrained(*args, **kwargs)
        super().__init__(fcn)


class TercilesFCNPostProcessing(DistributionModelAdapter):
    """FCN Post Processing model that outputs terciles."""

    def __init__(self, *args, **kwargs):
        fcn = DistributionFCNPostProcessing(*args, **kwargs)
        super().__init__(fcn)
