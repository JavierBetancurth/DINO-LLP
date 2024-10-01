import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.cluster import KMeans

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

    def forward(self, x):
        # Asegurarse de que prototypes está definido
        if self.prototypes is not None:
            prototypes = self.prototypes(x)
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

class PrototypeMemory(nn.Module):
    def __init__(self, num_prototypes, feat_dim, world_size, rank, dataset_size):
        super(PrototypeMemory, self).__init__()
        self.num_prototypes = num_prototypes
        self.feat_dim = feat_dim
        self.world_size = world_size
        self.rank = rank
        self.dataset_size = dataset_size
        
        # Memoria para guardar los embeddings de las vistas
        self.local_memory_embeddings = torch.zeros(dataset_size, feat_dim).cuda()
        self.local_memory_index = torch.zeros(dataset_size, dtype=torch.long).cuda()

    def update_memory(self, batch_embeddings, batch_indexes):
        self.local_memory_embeddings[batch_indexes] = batch_embeddings
        self.local_memory_index[batch_indexes] = batch_indexes

    def kmeans_clustering(self, num_iters=10):
        assignments = -100 * torch.ones(self.num_prototypes, self.dataset_size).long()
        j = 0
        for i_K, K in enumerate(self.num_prototypes):
            # Inicializar centroides con elementos de la memoria local
            centroids = torch.empty(K, self.feat_dim).cuda()
            if self.rank == 0:
                random_idx = torch.randperm(len(self.local_memory_embeddings[j]))[:K]
                centroids = self.local_memory_embeddings[j][random_idx]
            dist.broadcast(centroids, 0)

            for n_iter in range(num_iters):
                # E-step: calcular distancias a los centroides y asignar prototipos
                dot_products = torch.mm(self.local_memory_embeddings[j], centroids.t())
                _, local_assignments = dot_products.max(dim=1)

                # M-step: actualizar los centroides
                where_helper = self.get_indices_sparse(local_assignments.cpu().numpy())
                counts = torch.zeros(K).cuda().int()
                emb_sums = torch.zeros(K, self.feat_dim).cuda()
                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = torch.sum(self.local_memory_embeddings[j][where_helper[k][0]], dim=0)
                        counts[k] = len(where_helper[k][0])

                dist.all_reduce(counts)
                dist.all_reduce(emb_sums)

                mask = counts > 0
                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)
                centroids = nn.functional.normalize(centroids, dim=1, p=2)

            assignments[i_K] = local_assignments.cpu()
            j = (j + 1) % len(self.local_memory_embeddings)
        
        return assignments

    def get_indices_sparse(self, labels):
        where = {}
        for i, label in enumerate(labels):
            if label not in where:
                where[label] = []
            where[label].append(i)
        where = {k: (torch.LongTensor(v),) for k, v in where.items()}
        return where
