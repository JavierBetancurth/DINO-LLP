# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision #J
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead


# proportions
# from loss.kl_loss import compute_kl_loss_on_bagbatch
from proportions_assignments.prototypes_layer import Prototypes
from loss.koleo_loss import KoLeoLoss
from loss.koleo_loss_proportions import KoLeoLossProportions
from loss.llp_loss import ProportionLoss

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='dino_vits16', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup) 
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.32, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.32),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Sinkhorn Knopp parameters
    parser.add_argument("--epsilon", default=0.05, type=float,
       help="regularization parameter for Sinkhorn-Knopp algorithm")
    parser.add_argument("--n_iterations", default=3, type=int,
       help="number of iterations in Sinkhorn-Knopp algorithm")
    parser.add_argument("--nmb_prototypes", default=10, type=int,
       help="number of prototypes")

    # losses parameters
    parser.add_argument('--alpha', type=float, default=0.8, help="""alpha parameter defined to 
        weight between dino and kl losses.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    labels = dataset.targets
    
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ building prototype layer ... ============
    prototypes_layer = Prototypes(output_dim=args.out_dim, nmb_prototypes=args.nmb_prototypes).cuda()

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # Koleo loss
    koLeo_loss_fn = KoLeoLoss()
    # Proportion loss
    proportion_loss_fn = ProportionLoss(metric="l1", alpha=args.alpha)
    
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, dataset, prototypes_layer, koLeo_loss_fn, proportion_loss_fn, args)   # se agrega la variable dataset, prototypes_layer, proportion_loss_fn y koLeo_loss_fn

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# Calcular las proporciones de cada lote
def calculate_class_proportions_in_batch(labels, dataset):
    labels_tensor = labels.clone().detach().cuda() # Usar directamente y mover a CUDA
    class_counts = torch.bincount(labels_tensor, minlength=len(dataset.classes))
    class_proportions = class_counts.float() / len(labels_tensor)
    
    return class_proportions

def calculate_class_proportions_in_dataset(dataset):
    # Obtener todas las etiquetas del dataset
    all_labels = torch.tensor(dataset.targets, dtype=torch.long, device='cuda') # Usar directamente y mover a CUDA
    # Contar el número de instancias por clase
    class_counts = torch.bincount(all_labels, minlength=len(dataset.classes))
    # Calcular las proporciones globales
    total_samples = len(all_labels)
    class_proportions = class_counts.float() / total_samples

    return class_proportions


def compute_kl_loss_on_bagbatch(estimated_proportions, class_proportions, epsilon=1e-8):
    if not isinstance(class_proportions, torch.Tensor):
        real_proportions = torch.tensor(class_proportions, dtype=torch.float32).cuda()
    else:
        real_proportions = class_proportions

    # Asegurarse de que estimated_proportions también esté en la misma GPU
    # estimated_proportions = estimated_proportions.cuda() if not estimated_proportions.is_cuda else estimated_proportions

    # Forzar la normalización manualmente 
    # estimated_proportions /= estimated_proportions.sum(dim=-1, keepdim=True)

    '''
    # Si se utiliza la salida de la capa de prototipos
    # Calcular las probabilidades y la pérdida KL
    probabilities = F.softmax(estimated_proportions, dim=-1)
    avg_prob = torch.mean(probabilities, dim=0)
    avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)
    '''
    # Si se utiliza el algoritmo del Sinkhorn Knopp
    avg_prob = torch.mean(estimated_proportions, dim=0)
    avg_prob = torch.clamp(avg_prob, epsilon, 1 - epsilon)
    
    # Calcular diferencias entre proporciones reales y estimadas
    # differences = torch.abs(avg_prob - real_proportions)

    # Ponderar la pérdida KL con base en las diferencias
    loss = torch.sum(-real_proportions * torch.log(avg_prob), dim=-1)
    
    # Aplicar ponderación basada en las diferencias
    # weighted_loss = loss * torch.exp(differences)  # Escalar la pérdida según las diferencias

    # Ignorar las clases con proporciones reales de cero
    # mask = real_proportions > 0 # [mask]
    
    # Calcular la pérdida KL utilizando las proporciones del lote
    loss = torch.sum(-real_proportions * torch.log(avg_prob), dim=-1).mean()
    
    return loss # weighted_loss.mean() # loss

def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, dataset, prototypes_layer, koLeo_loss_fn, proportion_loss_fn, args):  # se agrega la variable dataset, prototypes_layer, proportion_loss_fn y koLeo_loss_fn
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)

    # Calcular proporciones globales del dataset
    class_proportions_global = calculate_class_proportions_in_dataset(dataset)
                        
    for it, (images, labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        
        # Calcular las proporciones de clase en el lote actual
        class_proportions = calculate_class_proportions_in_batch(labels, dataset)
        
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16_scaler is not None):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss1 = dino_loss(student_output, teacher_output, epoch)

            # Paso a través de la capa de Prototipos
            prototypes = prototypes_layer(student_output)
                
            # Normalizar los prototipos antes de Sinkhorn-Knopp
            prototypes = nn.functional.normalize(prototypes, dim=1, p=2)
                
            # Asignar recortes a prototipos con Sinkhorn-Knopp
            prototypes_output = sinkhorn_knopp(prototypes, temp=args.epsilon, n_iterations=args.n_iterations)
            
            # Asignar cada recorte a una clase (máxima probabilidad)
            recorte_asignaciones = torch.argmax(prototypes_output, dim=1) # (640,)
            # Calcular las proporciones observadas en el lote
            num_classes = 10  # Número de clases
            proporciones_observadas = torch.bincount(recorte_asignaciones, minlength=num_classes).float()
            proporciones_observadas /= recorte_asignaciones.size(0)  # Dividir por el número total de recortes
            
            # Convertir prototypes_output a proporciones reales y calcular la pérdida KL
            # loss2 = compute_kl_loss_on_bagbatch(prototypes_output, class_proportions_global, epsilon=1e-8)
            # Calcula las pérdidas
            # batch_proportion_prediction = prototypes_output.mean(dim=0)  # Promediar sobre todos los recortes (640, 10)
            loss2 = proportion_loss_fn(proporciones_observadas, class_proportions)

            # Calcula la pérdida KoLeo
            loss3 = koLeo_loss_fn(student_output)

            # Combinar las pérdidas usando el parámetro alpha
            # loss = args.alpha * loss1 + (1 - args.alpha) * loss2
    
            # Incrementa el peso de la pérdida KL
            loss = loss1 + args.alpha * loss2 +  loss3

        # Logging para monitorizar
        # print(f"Batch {it} - Proporciones reales: {class_proportions}")
        # print(f"Batch {it} - Proporciones estimadas: {torch.mean(prototypes_output, dim=0).cpu().numpy()}")
        # print(f"Batch {it} - Pérdida DINO: {loss1.item()}, Pérdida KL: {loss2.item()}, Pérdida Total: {loss.item()}")

        # imprimir información de las salidas (solo una vez)
        if it == 0 and utils.is_main_process():
            print("teacher output shape:", teacher_output.shape)
            print("teacher output type:", teacher_output.dtype)
            print("student output shape:", student_output.shape)
            print("student output type:", student_output.dtype)

            print("Teacher output shape:", teacher_output.shape)
            print("Student output shape:", student_output.shape)
            print("Prototypes output shape:", prototypes_output.shape)
            print("Class proportions shape:", class_proportions_global.shape)
            print("Loss1:", loss1.item())
            print("Loss2:", loss2.item())
            print("Loss3:", loss3.item())


        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        '''
        # **Monitoreo de los prototipos**
        if it % 50 == 0:  # Imprimir cada 50 iteraciones
            with torch.no_grad():
                print(f"Iteración {it} - Prototipos actuales (primeras 5 filas): {prototypes_layer.prototypes.weight[:5].cpu().numpy()}")
                
            # Monitorear gradientes de los prototipos
            if prototypes_layer.prototypes.weight.grad is not None:
                print(f"Gradientes de prototipos en iteración {it} (primeras 5 filas): {prototypes_layer.prototypes.weight.grad[:5].cpu().numpy()}")
            else:
                print(f"Gradientes de prototipos en iteración {it}: No se están actualizando los gradientes.")
        '''

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_dino=loss1.item())
        metric_logger.update(loss_kl=loss2.item())
        metric_logger.update(loss_koleo=loss3.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        # metric_logger.update(alpha=alpha)
        # metric_logger.update(accuracy=accuracy)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    '''
    # Imprimir proporciones estimadas y las proporciones reales al final de cada epoca
    # Calcular proporciones globales del dataset
    class_proportions_global = calculate_class_proportions_in_dataset(dataset)
    print("Proporciones de clases reales dataset:", class_proportions_global)                 
    real_proportions = class_proportions.clone().detach()  
    print("Proporciones de clases reales lote:", real_proportions)
    avg_estimated_proportions = torch.mean(prototypes_output, dim=0).clone().detach()
    print("Proporciones promedio estimadas:", avg_estimated_proportions)
    '''
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.printed_info = False  # Flag to ensure printing only once

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        # impresiones de la salidas función de pérdida
        if not self.printed_info:
            self.printed_info = True
            # Print information about teacher_out and student_out recrops
            for i, crop in enumerate(teacher_out):
                print(f"teacher output función de pérdida crop {i + 1} shape: {crop.shape}")
                print(f"teacher output función de pérdida crop {i + 1} type: {crop.dtype}")
            for i, crop in enumerate(student_out):
                print(f"student output función de pérdida crop {i + 1} shape: {crop.shape}")
                print(f"student output función de pérdida crop {i + 1} type: {crop.dtype}")

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
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

@torch.no_grad()
def sinkhorn_knopp(prototypes, temp, n_iterations):
        prototypes = prototypes.float()
        # print(prototypes.shape)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(prototypes / temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # Number of samples to assign
        K = Q.shape[0]  # How many prototypes
        # wi = wi.cuda()
        # print(K)

        # Make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # Normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # Normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()  
    
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3441, 0.3801, 0.4076), (0.2023, 0.1366, 0.1153)),       
        ]) # Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) 

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
