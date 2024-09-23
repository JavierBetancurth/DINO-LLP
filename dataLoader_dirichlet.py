import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import random

# Función para generar proporciones Dirichlet por lote
def generate_dirichlet_batch(labels, alpha, batch_size, num_classes):
    """
    Genera un lote basado en la distribución de Dirichlet.
    
    labels: las etiquetas del dataset
    alpha: parámetro de Dirichlet
    batch_size: tamaño del lote
    num_classes: número de clases en el dataset
    """
    # Obtener los índices por clase
    label_indices = {i: np.where(np.array(labels) == i)[0] for i in range(num_classes)}
    
    # Generar proporciones Dirichlet
    class_proportions = np.random.dirichlet([alpha] * num_classes, size=1).flatten()
    
    # Calcular el número de muestras por clase en el lote
    samples_per_class = (class_proportions * batch_size).astype(int)
    
    # Asegurar que el número total de muestras sea igual a batch_size
    samples_per_class[-1] += batch_size - sum(samples_per_class)
    
    selected_indices = []
    for i, indices in label_indices.items():
        # Selecciona muestras únicas para cada clase en el lote
        if len(indices) > 0:
            num_samples = min(samples_per_class[i], len(indices))
            selected_indices.extend(np.random.choice(indices, num_samples, replace=False))
    
    return selected_indices

# Dataset personalizado para asegurarnos de que los lotes cumplan con el criterio
class DirichletBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size, alpha, num_classes):
        self.labels = labels
        self.batch_size = batch_size
        self.alpha = alpha
        self.num_classes = num_classes

    def __iter__(self):
        # Generar lotes de manera iterativa
        while True:
            selected_indices = generate_dirichlet_batch(self.labels, self.alpha, self.batch_size, self.num_classes)
            yield selected_indices

    def __len__(self):
        return len(self.labels) // self.batch_size

# Función para crear el DataLoader con muestreo Dirichlet
def create_dataloader_dirichlet(data_path, transform, batch_size, num_workers, alpha, num_classes):
    # Cargar el dataset
    dataset = datasets.ImageFolder(data_path, transform=transform)
    
    # Obtener las etiquetas
    labels = [sample[1] for sample in dataset.samples]
    
    # Crear un sampler personalizado para los lotes Dirichlet
    dirichlet_sampler = DirichletBatchSampler(labels, batch_size, alpha, num_classes)
    
    # Crear el DataLoader
    data_loader = DataLoader(
        dataset,
        batch_sampler=dirichlet_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data loaded: {len(dataset)} images in total.")
    return data_loader

'''
# ============ Uso de la función ============

# Definir transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes
    transforms.ToTensor(),          # Convertir las imágenes a tensores
])

# Ruta al dataset
data_path = 

# Parámetros para el DataLoader
batch_size = 32
num_workers = 4
alpha = 0.5  # Parámetro de Dirichlet
num_classes = 10  # Asume que tienes 10 clases en el dataset

# Crear el DataLoader
data_loader = create_dataloader_dirichlet(
    data_path=data_path,
    transform=transform,
    batch_size=batch_size,
    num_workers=num_workers,
    alpha=alpha,
    num_classes=num_classes
)

# Prueba del DataLoader
for batch in data_loader:
    inputs, targets = batch
    print(f"Batch size: {len(inputs)}")
    break  # Solo para verificar el primer lote
'''
