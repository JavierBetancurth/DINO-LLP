"""
{citas}
"""
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from proportions_assignments.prototypes_layer import Prototypes

class DINOLossdFPMm(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9
      ):

        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    @torch.no_grad()
    def forward(self, 
                student_output, 
                teacher_output, 
                real_proportions, 
                estimated_proportions, 
                epoch, 
                alpha=0.5,
                beta=0.5,
               ):
                   
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

    # teacher_out_prototypes = Prototypes(teacher_output, numb_prototypes)
    
    @torch.no_grad()
    def sinkhorn_knopp_proportions(self, teacher_output, epsilon, n_iterations):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # apply the constraint to the transportation polytope
        constraint_matrix = torch.ones(K, B) * real_proportions.view(-1, 1)
        Q *= constraint_matrix

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

        # q = distributed_sinkhorn(student_out)
        
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # skip cases where student and teacher operate on the same view
                    continue
                # Original self-distillation loss
                loss1 = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)

                # Label proportions-based loss with symmetric cross entropy
                probabilities = nn.functional.softmax(estimated_proportions, dim=-1)
                avg_prob = torch.mean(probabilities, dim=0)
                avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)
                ce_loss = torch.sum(-real_proportions * F.log_softmax(avg_prob), dim=-1)
                rce_loss = torch.sum(-avg_prob, dim=-1) * torch.log(real_proportions)
                loss2 = ce_loss + beta * rce_loss

                # Combine the losses using the alpha parameter
                loss = alpha * loss1 + (1 - alpha) * loss2

                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
