import torch
import torch.nn as nn

# Label proportions-based loss with asymmetric cross entropy
def compute_kl_loss_on_bagbatch(estimated_proportions, class_proportions_list, device_ids=[args.gpu], epsilon=1e-8):
    for i in range(len(class_proportions_list)):
        real_proportions = class_proportions_list[i]  # Proporciones reales del lote actual
        # Move tensors to the configured device
        estimated_proportions = estimated_proportions.to(device)
        real_proportions = real_proportions.to(device)
            
        # Calcular las probabilidades y la pérdida KL
        probabilities = nn.functional.softmax(estimated_proportions, dim=-1)
        avg_prob = torch.mean(probabilities, dim=0)
        avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)
            
        # Calcular la pérdida KL utilizando las proporciones del lote
        loss = torch.sum(-real_proportions * torch.log(avg_prob), dim=-1).mean()
        
    return loss


# ce_loss = torch.sum(-real_proportions * F.log_softmax(estimated_proportions), dim=-1)
# rce_loss = torch.sum(-estimated_proportions, dim=-1) * torch.log(real_proportions)
# loss2 = ce_loss + beta * rce_loss
                    
