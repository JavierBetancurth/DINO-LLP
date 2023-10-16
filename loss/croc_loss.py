class CrOCLoss(nn.Module):
    def __init__(self, out_dim, out_dim_c, ncrops, warmup_teacher_temp, warmup_teacher_temp_c, teacher_temp, teacher_temp_c,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, student_temp_c=0.1, center_momentum=0.9, center_momentum_c=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.student_temp_c = student_temp_c
        self.center_momentum = center_momentum
        self.center_momentum_c = center_momentum_c
        self.ncrops = ncrops
        self.centroids_counter = torch.tensor(0, device='cuda')
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center_c", torch.zeros(1, out_dim_c))
        # we apply a warm-up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.teacher_temp_schedule_c = np.concatenate((
            np.linspace(warmup_teacher_temp_c, teacher_temp_c, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp_c
        ))

        # Log metrics
        self.teacher_entropy = torch.tensor(0, device='cuda')
        self.teacher_entropy_c = torch.tensor(0, device='cuda')
        self.kl_div = torch.tensor(0, device='cuda')
        self.kl_div_c = torch.tensor(0, device='cuda')

    @torch.no_grad()
    def compute_metrics_dino(self, teacher_out, student_out):
        # Compute the teacher's entropy
        self.teacher_entropy = Categorical(probs=teacher_out).entropy().mean()
        dist.all_reduce(self.teacher_entropy)
        self.teacher_entropy = self.teacher_entropy / dist.get_world_size()

        # Compute the KL divergence
        self.kl_div = -torch.nn.KLDivLoss(reduction='batchmean')(student_out, teacher_out)

    @torch.no_grad()
    def compute_metrics_croc(self, student_cent, teacher_cent):
        # Compute the teacher's entropy
        self.teacher_entropy_c = Categorical(probs=teacher_cent).entropy().sum()
        dist.all_reduce(self.teacher_entropy_c)
        self.teacher_entropy_c = self.teacher_entropy_c / self.centroids_counter

        # Compute the KL divergence
        student_cent_v1, student_cent_v2 = student_cent.chunk(2)
        teacher_cent_v1, teacher_cent_v2 = teacher_cent.chunk(2)
        kl_div_1 = torch.nn.KLDivLoss(reduction='batchmean')(student_cent_v1, teacher_cent_v2)
        kl_div_2 = torch.nn.KLDivLoss(reduction='batchmean')(student_cent_v2, teacher_cent_v1)
        self.kl_div_c = -(kl_div_1 + kl_div_2) / 2.

    def forward(self, student_output, teacher_output, epoch, student_centroids=None, teacher_centroids=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

        # Compute the loss on the centroids if provided
        loss_c = torch.tensor(0.).to(student_output.device)
        if student_centroids is not None and teacher_centroids is not None:
            # Sharpen the student predictions
            student_cent = student_centroids / self.student_temp_c

            # Teacher centering and sharpening
            temp = self.teacher_temp_schedule_c[epoch]
            teacher_cent = F.softmax((teacher_centroids - self.center_c) / temp, dim=-1)

            # Split the centroids view-wise
            student_cent_v1, student_cent_v2 = student_cent.chunk(2)
            teacher_cent_v1, teacher_cent_v2 = teacher_cent.chunk(2)

            # Compute the loss
            loss_c += torch.sum(-teacher_cent_v1 * F.log_softmax(student_cent_v2, dim=-1), dim=-1).mean()
            loss_c += torch.sum(-teacher_cent_v2 * F.log_softmax(student_cent_v1, dim=-1), dim=-1).mean()
            loss_c /= 2.

        # Update the centers
        self.update_center(teacher_output, teacher_centroids)

        # Update the metrics
        self.compute_metrics_dino(torch.cat(teacher_out), torch.cat(student_out))
        if student_centroids is not None and teacher_centroids is not None:
            self.compute_metrics_croc(student_cent, teacher_cent)
        return total_loss, loss_c

    @torch.no_grad()
    def update_center(self, teacher_output, teacher_centroids=None):
        """
        Update center used for teacher output.
        """
        # Image-level
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)

        # Update
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        # Centroids-level
        if teacher_centroids is not None:
            batch_center_c = torch.sum(teacher_centroids, dim=0, keepdim=True)
            self.centroids_counter = torch.tensor(len(teacher_centroids), device='cuda')

            # Update
            dist.all_reduce(batch_center_c)
            dist.all_reduce(self.centroids_counter)
            batch_center_c = batch_center_c / self.centroids_counter
            self.center_c = self.center_c * self.center_momentum_c + batch_center_c * (1 - self.center_momentum_c)
