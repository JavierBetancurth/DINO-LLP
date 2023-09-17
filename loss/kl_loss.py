import torch
import torch.nn as nn

# Label proportions-based loss with symmetric cross entropy
def compute_kl_loss_on_bagbatch(estimated_proportions, real_proportions, device, epsilon=1e-8):
    # Move tensors to the configured device
    estimated_proportions = estimated_proportions.to(device)
    real_proportions = real_proportions.to(device)
    # Forward pass
    batch_size, real_proportions = 
    outputs = teacher_out_prototypes
    probabilities = nn.functional.softmax(outputs, dim=-1).reshape((batch_size, bag_size, -1))
    avg_prob = torch.mean(probabilities, dim=1)
    avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)
    loss = torch.sum(-probabilities * torch.log(avg_prob), dim=-1).mean()

    return loss


    # ce_loss = torch.sum(-real_proportions * F.log_softmax(estimated_proportions), dim=-1)
    # rce_loss = torch.sum(-estimated_proportions, dim=-1) * torch.log(real_proportions)
    # loss2 = ce_loss + beta * rce_loss
                    
