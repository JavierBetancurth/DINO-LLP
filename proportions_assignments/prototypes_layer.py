import torch
import torch.nn as nn

class Prototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes=0, init_weights=True):
        super(Prototypes, self).__init__()

        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        if init_weights:
            self._initialize_weights()

    def forward(self, output_dim):
        if self.prototypes is not None:
            prototypes = self.prototypes(output_dim)
        else:
            prototypes = self.prototype_layer(output_dim)
        return prototypes

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes_list):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes_list)
        for i, nmb_prototypes in enumerate(nmb_prototypes_list):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, nmb_prototypes, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out