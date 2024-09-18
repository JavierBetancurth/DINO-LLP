'''
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
                nn.init.constant_(m.bias, 0) es necxesario hacer esto # Convertir prototypes_output a proporciones reales y calcular la pérdida KL
            prototypes_proportions = torch.sum(prototypes_output, dim=0) / torch.sum(prototypes_output)
'''

class Prototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes=0, init_weights=True):
        super(Prototypes, self).__init__()

        self.prototypes = None
        # Si nmb_prototypes es una lista, usaremos MultiPrototypes
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        # Si nmb_prototypes es mayor que 0, usamos una capa lineal
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
        # Si nmb_prototypes es 0, podemos manejar este caso con una excepción o asignación especial
        else:
            raise ValueError("El número de prototipos debe ser mayor que 0 o una lista.")

        if init_weights:
            self._initialize_weights()

    def forward(self, output_dim):
        # Asegurarse de que prototypes está definido
        if self.prototypes is not None:
            prototypes = self.prototypes(output_dim)
        else:
            raise ValueError("La capa de prototipos no está definida correctamente.")
        return prototypes

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
