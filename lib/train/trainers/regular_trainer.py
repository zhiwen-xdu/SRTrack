import os
import datetime
from collections import OrderedDict
from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue
from lib.train.admin import TensorboardWriter
import torch
import time
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from lib.utils.misc import get_world_size


class RegularTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None, use_amp=False):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()

        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
            if not os.path.exists(tensorboard_writer_dir):
                os.makedirs(tensorboard_writer_dir)
            self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])

            if settings.use_wandb:
                world_size = get_world_size()
                cur_train_samples = self.loaders[0].dataset.samples_per_epoch * max(0, self.epoch - 1)
                interval = (world_size * settings.batchsize)  # * interval

        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
        self.settings = settings
        self.use_amp = use_amp
        if use_amp:
            self.scaler = GradScaler()

        # Momentum Factor
        self.m_min = 0.25
        self.m_max = 0.65
        # ====For _momentum_update_with_sensitivity=====
        self.grads_keep = {n: 0 * p.clone() for n, p in self.actor.net.backbone.named_parameters() if p.requires_grad}


    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)

    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        '''add fix rgb pretrained net bn, only used in box_head'''
        if self.settings.fix_bn:
            self.actor.fix_bns()

        self._init_timing()

        for i, data in enumerate(loader, 1):
            self.data_read_done_time = time.time()
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            self.data_to_gpu_time = time.time()

            data['epoch'] = self.epoch
            data['settings'] = self.settings
            # forward pass
            if not self.use_amp:
                loss, stats = self.actor(data)
            else:
                with autocast():
                    loss, stats = self.actor(data)

            # For Momentum Update
            # ====For _momentum_update=====
            # params_keep = [p.clone() for n,p in self.actor.net.backbone.named_parameters() if p.requires_grad]
            # ====For _momentum_update_with_sensitivity=====
            params_keep = {n: p.clone() for n,p in self.actor.net.backbone.named_parameters() if p.requires_grad}
            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                if not self.use_amp:
                    loss.backward()
                    if self.settings.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.actor.net.parameters(), self.settings.grad_clip_norm)
                    self.optimizer.step()
                    # self._momentum_update(params_keep)
                    self._momentum_update_with_sensitivity(params_keep)
                    # self._update_with_sensitivity(params_keep)
                else:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            # update statistics
            batch_size = data['template_images'].shape[loader.stack_dim]
            self._update_stats(stats, batch_size, loader)

            # print statistics
            self._print_stats(i, loader, batch_size)

            torch.cuda.empty_cache()

        # calculate ETA after every epoch
        epoch_time = self.prev_time - self.start_time
        print("Epoch Time: " + str(datetime.timedelta(seconds=epoch_time)))
        print("Avg Data Time: %.5f" % (self.avg_date_time / self.num_frames * batch_size))
        print("Avg GPU Trans Time: %.5f" % (self.avg_gpu_trans_time / self.num_frames * batch_size))
        print("Avg Forward Time: %.5f" % (self.avg_forward_time / self.num_frames * batch_size))


    @torch.no_grad()
    def _momentum_update(self, params_keep):
        """ Momentum update of the encoder """
        params_tuned = [p for n, p in self.actor.net.backbone.named_parameters() if p.requires_grad]
        for param_moco, param_keep in zip(params_tuned, params_keep):
            param_moco.data = param_keep.data * self.m + param_moco.data * (1.0 - self.m)

    # =====================1=====================
    # def _momentum_update_with_sensitivity(self, params_keep):
    #        """Get the sensitivity and the trainable parameter configurations."""
    #        # Pre-defined keywords for not calculating sensitivity
    #        grad_skip_list = ['patch_embed', 'norm', 'bias']  # 'patch_embed', 'norm_rgb', 'norm_event', 'norm1', 'norm2', 'bias'
    #
    #        # Accumulating gradient for a epoch
    #        for n, p in self.actor.net.backbone.named_parameters():
    #            if p.requires_grad and not any(k in n for k in grad_skip_list):
    #                # Sensitivity
    #                sensitivities = (p.grad ** 2).detach() * 1e8
    #                sensitivities = sensitivities.flatten()
    #                # Sensitivity to Momentum Factor
    #                _, sort_id = sensitivities.sort(descending=True)  # 从大到小
    #                momentums = torch.linspace(self.m, 0.99, steps=sensitivities.shape[0])
    #                momentums = momentums.to(sensitivities.device)
    #                sort_momentums = momentums[sort_id]
    #                sort_momentums = sort_momentums.view(p.grad.shape)
    #                # Momentum Updata
    #                p_keep = params_keep[n]
    #                p.data = p_keep.data * sort_momentums + p.data * (1.0 - sort_momentums)
    #
    #            elif p.requires_grad and any(k in n for k in grad_skip_list):
    #                # No Sensitivity
    #                sort_momentums = self.m * torch.ones_like(p.grad, device=p.grad.device)
    #                # Momentum Updata
    #                p_keep = params_keep[n]
    #                p.data = p_keep.data * sort_momentums + p.data * (1.0 - sort_momentums)

    # =====================2=====================
    # def _momentum_update_with_sensitivity(self, params_keep):
    #        """Get the sensitivity and the trainable parameter configurations."""
    #        # Accumulating gradient for a epoch
    #        for n, p in self.actor.net.backbone.named_parameters():
    #            if p.requires_grad:
    #                # Sensitivity
    #                sensitivities = (p.grad ** 2).detach() * 1e8
    #                sensitivities = sensitivities.flatten()
    #                # Sensitivity to Momentum Factor
    #                _, sort_id = sensitivities.sort(descending=True)  # 从大到小
    #                momentums = torch.linspace(self.m, self.m_max, steps=sensitivities.shape[0])
    #                momentums = momentums.to(sensitivities.device)
    #                sort_momentums = momentums[sort_id]
    #                sort_momentums = sort_momentums.view(p.grad.shape)
    #                # Momentum Updata
    #                p_keep = params_keep[n]
    #                p.data = p_keep.data * sort_momentums + p.data * (1.0 - sort_momentums)

    # =====================3=====================
    # def _momentum_update_with_sensitivity(self, params_keep):
    #        """Get the sensitivity and the trainable parameter configurations."""
    #        # Accumulating gradient for a epoch
    #        for n, p in self.actor.net.backbone.named_parameters():
    #            if p.requires_grad:
    #                # Sensitivity
    #                sensitivities = (p.grad ** 2).detach() * 1e8
    #                sensitivities = sensitivities.flatten()
    #                # Sensitivity to Momentum Factor
    #                _, sort_id = sensitivities.sort(descending=True)  # 从大到小
    #                momentums_1 = torch.linspace(self.m, self.m, steps=sensitivities.shape[0]//2)
    #                momentums_2 = torch.linspace(self.m, self.m_max, steps=sensitivities.shape[0]//2)  # 前一半[self.m,self.m], 后一半前一半[self.m,self.m_max],
    #                momentums = torch.cat([momentums_1,momentums_2],dim=0)
    #                momentums = momentums.to(sensitivities.device)
    #                sort_momentums = momentums[sort_id]
    #                sort_momentums = sort_momentums.view(p.grad.shape)
    #                # Momentum Updata
    #                p_keep = params_keep[n]
    #                p.data = p_keep.data * sort_momentums + p.data * (1.0 - sort_momentums)


    # # =====================4=====================
    # def _momentum_update_with_sensitivity(self, params_keep):
    #        """Get the sensitivity and the trainable parameter configurations."""
    #        # Accumulating gradient for a epoch
    #        for n, p in self.actor.net.backbone.named_parameters():
    #            if p.requires_grad:
    #                # Sensitivity
    #                sensitivities = (p.grad ** 2).detach() * 1e8
    #                sensitivities = sensitivities.flatten()
    #                # Sensitivity to Momentum Factor
    #                _, sort_id = sensitivities.sort(descending=False)  # 从小到大
    #                momentums = torch.linspace(self.m, self.m_max, steps=sensitivities.shape[0])
    #                momentums = momentums.to(sensitivities.device)
    #                sort_momentums = momentums[sort_id]
    #                sort_momentums = sort_momentums.view(p.grad.shape)
    #                # Momentum Updata
    #                p_keep = params_keep[n]
    #                p.data = p_keep.data * sort_momentums + p.data * (1.0 - sort_momentums)


    # =====================5=====================
    def _momentum_update_with_sensitivity(self, params_keep):
           """Get the sensitivity and the trainable parameter configurations."""
           # Accumulating gradient for a epoch
           for n, p in self.actor.net.backbone.named_parameters():
               if p.requires_grad:
                   # ===(1) Grad to Sensitivity===
                   grad_square_keep = self.grads_keep[n]
                   grad_square_curr = (p.grad ** 2).detach() * 1e8
                   # Sensitivity 也用Momentum计算
                   grad_square = (grad_square_keep + grad_square_curr)/2.0
                   # 用于记录前一步的grads
                   self.grads_keep[n] = grad_square_curr
                   # Normalization Grad
                   # print("==1==",n,grad_square.shape,torch.max(grad_square))
                   sensitivities = grad_square / torch.max(grad_square)
                   sensitivities = sensitivities.flatten()
                   # ===(2) Sensitivity to Momentum Factor===
                   _, sort_id = sensitivities.sort(descending=False)  # 从小到大
                   momentums = torch.linspace(self.m_min, self.m_max, steps=sensitivities.shape[0])
                   momentums = momentums.to(sensitivities.device)
                   sort_momentums = momentums[sort_id]
                   sort_momentums = sort_momentums.view(p.grad.shape)
                   if "blocks_event.4.norm1.weight" in n:
                       print(float(sort_momentums[0]))

                   # Momentum Updata
                   p_keep = params_keep[n]
                   p.data = p_keep.data * sort_momentums + p.data * (1.0 - sort_momentums)


    # =====================6=====================
    # def _momentum_update_with_sensitivity(self, params_keep):
    #        """Get the sensitivity and the trainable parameter configurations."""
    #        momentum_list = torch.linspace(self.m_max, self.m_min, steps=12)  # 12 feature blocks
    #        most_sensitive_params = ["norm_rgb.", "norm_event.", "patch_embed_rgb.", "patch_embed_event."]
    #        block_params = ["." + str(i) + "." for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    #        for n, p in self.actor.net.backbone.named_parameters():
    #            if p.requires_grad:
    #                # ===(1) Grad to Sensitivity===
    #                for idx,p_name in enumerate(block_params):
    #                    if p_name in n:
    #                        momentum = momentum_list[idx]
    #                for p_name in most_sensitive_params:
    #                    if p_name in n:
    #                        momentum = self.m_max
    #
    #                # ===(2) Momentum Updata ===
    #                p_keep = params_keep[n]
    #                p.data = p_keep.data * momentum + p.data * (1.0 - momentum)


    # # =====================7=====================
    # def _momentum_update_with_sensitivity(self, params_keep):
    #     """Get the sensitivity and the trainable parameter configurations."""
    #     cross_momentum_list = torch.linspace(self.m_max, self.m_min, steps=4)  # 4 cross blocks
    #     momentum_list = torch.linspace(self.m_max, self.m_min, steps=12)  # 12 feature blocks
    #     most_sensitive_params = ["norm_rgb.", "norm_event.", "patch_embed_rgb.", "patch_embed_event."]
    #     block_params = ["." + str(i) + "." for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    #     cross_block_params = ["cross." + str(i) + "." for i in [0, 1, 2, 3]]
    #     for n, p in self.actor.net.backbone.named_parameters():
    #         if p.requires_grad:
    #             # ===(1) Grad to Sensitivity===
    #             for idx, p_name in enumerate(block_params):
    #                 if p_name in n:
    #                     momentum = momentum_list[idx]
    #             for p_name in most_sensitive_params:
    #                 if p_name in n:
    #                     momentum = self.m_max
    #             for idx, p_name in enumerate(cross_block_params):
    #                 if p_name in n:
    #                     momentum = cross_momentum_list[idx]
    #
    #             # ===(2) Momentum Updata ===
    #             p_keep = params_keep[n]
    #             p.data = p_keep.data * momentum + p.data * (1.0 - momentum)


    # =====================8=====================
    # def _momentum_update_with_sensitivity(self, params_keep):
    #     """Get the sensitivity and the trainable parameter configurations."""
    #     momentum_list = torch.linspace(self.m_max, self.m_min, steps=12)  # 12 feature blocks
    #     most_sensitive_params = ["norm_rgb.", "norm_event.", "patch_embed_rgb.", "patch_embed_event."]
    #     block_params = ["." + str(i) + "." for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    #     cross_block_params = ["cross." + str(i) + "." for i in [0, 1, 2, 3]]
    #     for n, p in self.actor.net.backbone.named_parameters():
    #         if p.requires_grad:
    #             # ===(1) Grad to Sensitivity===
    #             for idx, p_name in enumerate(block_params):
    #                 if p_name in n:
    #                     momentum = momentum_list[idx]
    #             for p_name in most_sensitive_params:
    #                 if p_name in n:
    #                     momentum = self.m_max
    #             for idx, p_name in enumerate(cross_block_params):
    #                 if p_name in n:
    #                     momentum = momentum_list[idx*3+2]
    #
    #             # ===(2) Momentum Updata ===
    #             p_keep = params_keep[n]
    #             p.data = p_keep.data * momentum + p.data * (1.0 - momentum)

    #
    # def _update_with_sensitivity(self, params_keep):
    #        """Get the sensitivity and the trainable parameter configurations."""
    #        # Accumulating gradient for a epoch
    #        print("===================================================================================================================================================")
    #        for n, p in self.actor.net.backbone.named_parameters():
    #            if p.requires_grad:
    #                # ===(1) Grad to Sensitivity===
    #                grad_square_keep = self.grads_keep[n]
    #                grad_square_curr = (p.grad ** 2).detach() * 1e8
    #                # Sensitivity 也用Momentum计算
    #                grad_square = grad_square_keep + grad_square_curr
    #                self.grads_keep[n] = grad_square
    #                # Sensitivity 也用Momentum计算
    #                if "qkv.weight" in n and "event" in n and "bias" not in n:
    #                    v_list = []
    #                    v = grad_square[1536:,:]  # [768,768]
    #                    for i in range(24):
    #                        v_slice = v[:,i*32:i*32+32]
    #                        v_list.append(float(torch.mean(v_slice)))
    #                    print(v_list,",")


    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                # 2021.1.10 Set epoch
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.avg_date_time = 0
        self.avg_gpu_trans_time = 0
        self.avg_forward_time = 0

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        # add lr state
        if loader.training:
            lr_list = self.lr_scheduler.get_last_lr()
            for i, lr in enumerate(lr_list):
                var_name = 'LearningRate/group{}'.format(i)
                if var_name not in self.stats[loader.name].keys():
                    self.stats[loader.name][var_name] = StatValue()
                self.stats[loader.name][var_name].update(lr)

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        prev_frame_time_backup = self.prev_time
        self.prev_time = current_time

        self.avg_date_time += (self.data_read_done_time - prev_frame_time_backup)
        self.avg_gpu_trans_time += (self.data_to_gpu_time - self.data_read_done_time)
        self.avg_forward_time += current_time - self.data_to_gpu_time

        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)

            # 2021.12.14 add data time print
            print_str += 'DataTime: %.3f (%.3f)  ,  ' % (self.avg_date_time / self.num_frames * batch_size, self.avg_gpu_trans_time / self.num_frames * batch_size)
            print_str += 'ForwardTime: %.3f  ,  ' % (self.avg_forward_time / self.num_frames * batch_size)
            print_str += 'TotalTime: %.3f  ,  ' % ((current_time - self.start_time) / self.num_frames * batch_size)

            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats):
                    if hasattr(val, 'avg'):
                        print_str += '%s: %.5f  ,  ' % (name, val.avg)

            print(print_str[:-5])
            log_str = print_str[:-5] + '\n'
            with open(self.settings.log_file, 'a') as f:
                f.write(log_str)

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                try:
                    lr_list = self.lr_scheduler.get_last_lr()
                except:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

