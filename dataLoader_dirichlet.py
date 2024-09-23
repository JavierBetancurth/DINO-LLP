import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets

# Función para generar proporciones Dirichlet
def generate_dirichlet_splits(labels, alpha, num_classes):
    """
    Genera un muestreo basado en la distribución de Dirichlet.
    
    labels: las etiquetas del dataset
    alpha: parámetro de Dirichlet
    num_classes: número de clases en el dataset
    """
    label_indices = {i: np.where(np.array(labels) == i)[0] for i in range(num_classes)}
    class_counts = [len(indices) for indices in label_indices.values()]
    
    # Genera proporciones Dirichlet para cada clase
    class_proportions = np.random.dirichlet([alpha] * num_classes, size=1).flatten()
    
    # Asigna el número de muestras por clase
    samples_per_class = (class_proportions * sum(class_counts)).astype(int)
    
    selected_indices = []
    for i, indices in label_indices.items():
        selected_indices.extend(np.random.choice(indices, samples_per_class[i], replace=False))
    
    return selected_indices

# Función del dataloader
def create_dataloader(data_path, transform, batch_size, num_workers, alpha, num_classes):
    # Cargar el dataset
    dataset = datasets.ImageFolder(data_path, transform=transform)
    
    # Obtener las etiquetas del dataset
    labels = dataset.targets
    
    # Generar los splits Dirichlet
    selected_indices = generate_dirichlet_splits(labels, alpha, num_classes)
    
    # Crear un sampler basado en los índices seleccionados
    sampler = SubsetRandomSampler(selected_indices)
    
    # Crear el DataLoader
    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"Data loaded: there are {len(selected_indices)} images sampled.")
    return data_loader

'''
# ============ Uso de la función ============

transform = DataAugmentationDINO(
    args.global_crops_scale,
    args.local_crops_scale,
    args.local_crops_number,
)

data_loader = create_dataloader(
    data_path=args.data_path,
    transform=transform,
    batch_size=args.batch_size_per_gpu,
    num_workers=args.num_workers,
    alpha=0.5,  # Parámtero de Dirichlet, ajustable
    num_classes=10  # Número de clases
)
'''
