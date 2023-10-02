import torch
import torch.nn as nn
import torch.nn.functional as F

# Label proportions-based loss with asymmetric cross entropy
def compute_kl_loss_on_bagbatch(estimated_proportions, class_proportions_list, epsilon=1e-8):
    estimated_proportions.clone().detach().requires_grad_(True) 
    # estimated_proportions = estimated_proportions.cuda().requires_grad_()
    # estimated_proportions = torch.tensor(estimated_proportions, dtype=torch.float32)

    for i in range(len(class_proportions_list)):
        real_proportions = class_proportions_list[i]  # Proporciones reales del lote actual

        # real_proportions.clone().detach().requires_grad_(True)
        real_proportions = torch.tensor(real_proportions, dtype=torch.float32).cuda()
        
        # Calcular las probabilidades y la pérdida KL
        probabilities = nn.functional.softmax(estimated_proportions, dim=-1)
        avg_prob = torch.mean(probabilities, dim=0)
        avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)
            
        # Calcular la pérdida KL utilizando las proporciones del lote
        loss = torch.sum(-real_proportions * torch.log(avg_prob), dim=-1).mean()
        
    return loss


# Label proportions-based loss with symmetric cross entropy
def compute_kl_loss_on_bagbatch2(estimated_proportions, class_proportions_list, epsilon=1e-8, beta=1.0):
    estimated_proportions.clone().detach().requires_grad_(True) 
    # estimated_proportions = estimated_proportions.cuda().requires_grad_()
    # estimated_proportions = torch.tensor(estimated_proportions, dtype=torch.float32)

    total_loss = 0
    n_loss_terms = 0

    for i in range(len(class_proportions_list)):
        real_proportions = class_proportions_list[i]  # Proporciones reales del lote actual

        # real_proportions.clone().detach().requires_grad_(True)
        real_proportions = torch.tensor(real_proportions, dtype=torch.float32).cuda()
        
        # Calcular las probabilidades y la pérdida KL
        # probabilities = nn.functional.softmax(estimated_proportions, dim=-1)
        probabilities = F.log_softmax(estimated_proportions, dim=-1)
        avg_prob = torch.mean(probabilities, dim=0)
        avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)

        # Calcular la pérdida KL utilizando las proporciones del lote
        ce_loss = torch.sum(-real_proportions * torch.log(avg_prob), dim=-1)
        rce_loss = torch.sum(-avg_prob * torch.log(real_proportions), dim=-1)
        loss = ce_loss + beta * rce_loss

        total_loss += loss.mean()
        n_loss_terms += 1
    total_loss /= n_loss_terms    
    return total_loss
        

# ce_loss = torch.sum(-real_proportions * F.log_softmax(estimated_proportions), dim=-1)
# rce_loss = torch.sum(-estimated_proportions, dim=-1) * torch.log(real_proportions)
# loss2 = ce_loss + beta * rce_loss
                    
