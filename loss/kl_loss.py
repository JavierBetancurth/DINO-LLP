import torch
import torch.nn as nn

# Label proportions-based loss with asymmetric cross entropy
def compute_kl_loss_on_bagbatch(estimated_proportions, class_proportions_list, epsilon=1e-8):
    estimated_proportions.clone().detach().requires_grad_(True) 
    # estimated_proportions = estimated_proportions.cuda().requires_grad_()
    # estimated_proportions = torch.tensor(estimated_proportions, dtype=torch.float32)

    for i in range(len(class_proportions_list)):
        real_proportions = class_proportions_list[i]  # Proporciones reales del lote actual

        #real_proportions.clone().detach().requires_grad_(True)
        real_proportions = torch.tensor(real_proportions, dtype=torch.float32).cuda()
        
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
                    
