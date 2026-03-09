import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
# from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
# from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset import VisEvent,FE108,CoeSot,DepthTrack,LasHeR
from lib.train.data import sampler, opencv_loader, processing, LTRLoader, sequence_sampler, SLTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.fix_bn = getattr(cfg.TRAIN, "FIX_BN", False) # add for fixing base model bn layer


def names2datasets(name_list: list, settings):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        if name == "VisEvent_Train":
            datasets.append(VisEvent(settings.env.visevent_dir, split='train'))
        elif name == "VisEvent_Val":
            datasets.append(VisEvent(settings.env.visevent_dir, split='val'))
        elif name == "FE108_Train":
            datasets.append(FE108(settings.env.fe108_dir, split='train'))
        elif name == "FE108_Val":
            datasets.append(FE108(settings.env.fe108_dir, split='val'))
        elif name == "CoeSot_Train":
            datasets.append(CoeSot(settings.env.coesot_dir, split='train'))
        elif name == "CoeSot_Val":
            datasets.append(CoeSot(settings.env.coesot_dir, split='val'))
        elif name == "LasHeR_Train":
            datasets.append(LasHeR(settings.env.lasher_dir, split='train'))
        elif name == "LasHeR_Val":
            datasets.append(LasHeR(settings.env.lasher_dir, split='val'))
        elif name == "DepthTrack_Train":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='train'))
        elif name == "DepthTrack_Val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='val'))
        elif name == "VisEvent_DepthTrack_LasHeR_Train":
            datasets.append(VisEvent(settings.env.visevent_dir, split='train'))
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='train'))
            datasets.append(LasHeR(settings.env.lasher_dir, split='train'))
        elif name == "DepthTrack_LasHeR_Val":
            datasets.append(DepthTrack(settings.env.depthtrack_dir, split='val'))
            datasets.append(LasHeR(settings.env.lasher_dir, split='val'))
    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    # Note: for multimodal data, ToGrayscale and Normalize need modify
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.ViPTProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.ViPTProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders(visevent no val split)
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings),
                                            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_val,
                                            frame_sample_mode=sampler_mode, train_cls=train_cls)
        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val



def build_seq_dataloaders(cfg, settings):
    # Data transform
    # Note: for multimodal data, ToGrayscale and Normalize need modify
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    # transform_joint = tfm.Transform(tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SeqProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)

    data_processing_val = processing.SeqProcessing(search_area_factor=search_area_factor,
                                                     output_sz=output_sz,
                                                     center_jitter_factor=settings.center_jitter_factor,
                                                     scale_jitter_factor=settings.scale_jitter_factor,
                                                     mode='sequence',
                                                     transform=transform_val,
                                                     joint_transform=transform_joint,
                                                     settings=settings)

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)

    dataset_train = sequence_sampler.SequenceSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, max_interval=cfg.DATA.MAX_INTERVAL,
                                            num_search_frames=cfg.DATA.SEARCH.NUMBER, num_template_frames=1, processing=data_processing_train,
                                            frame_sample_mode='sequential',prob=0.5)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=0, sampler=train_sampler)

    # Validation samplers and loaders(visevent no val split)
    if cfg.DATA.VAL.DATASETS_NAME[0] is None:
        loader_val = None
    else:
        dataset_val = sequence_sampler.SequenceSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings),
                                            p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, max_interval=cfg.DATA.MAX_INTERVAL,
                                            num_search_frames=settings.num_search, num_template_frames=1, processing=data_processing_val,
                                            frame_sample_mode='sequential',prob=0.5)

        val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
        loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=0, sampler=val_sampler,
                            epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val



def get_optimizer_scheduler(net, cfg):
    train_type = getattr(cfg.TRAIN.PROMPT, "TYPE", "")
    # ===== Fine-tune =====
    for n, p in net.named_parameters():
        p.requires_grad = False

    train_param_norm = ["backbone.norm_rgb.","backbone.norm_event."]
    train_param_embed_rgb =  ["backbone.patch_embed_rgb."]
    # train_param_vit_rgb = ["backbone.blocks_rgb." + str(i) + ".mlp" for i in [2,5,8,11]]
    # train_param_vit_rgb = ["backbone.blocks_rgb." + str(i) + ".mlp" for i in [1, 3, 5, 7, 9, 11]]
    # train_param_vit_rgb = ["backbone.blocks_rgb." + str(i) + ".mlp" for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    train_param_vit_rgb = ["backbone.blocks_rgb." + str(i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    train_param_embed_event =  ["backbone.patch_embed_event."]
    # train_param_vit_event = ["backbone.blocks_event." + str(i) + ".mlp" for i in [2,5,8,11]]
    # train_param_vit_event = ["backbone.blocks_event." + str(i) + ".mlp" for i in [1, 3, 5, 7, 9, 11]]
    # train_param_vit_event = ["backbone.blocks_event." + str(i) + ".mlp" for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    train_param_vit_event = ["backbone.blocks_event." + str(i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    # train_param_vit_cross = ["backbone.blocks_cross." + str(i) + ".mlp" for i in [0,1,2,3]]
    train_param_vit_cross = ["backbone.blocks_cross." + str(i) for i in [0,1,2,3]]
    # train_param_box_head = ["box_head"]

    train_param_norm_list = []
    train_param_embed_rgb_list = []
    train_param_embed_event_list = []
    train_param_vit_rgb_list = []
    train_param_vit_event_list = []
    train_param_vit_cross_list = []

    if 'vipt' in train_type:
        for n, p in net.named_parameters():
            for p_name in train_param_norm:
                if p_name in n:
                    p.requires_grad = True
                    train_param_norm_list.append(n)

            for p_name in train_param_embed_rgb:
                if p_name in n:
                    p.requires_grad = True
                    train_param_embed_rgb_list.append(n)

            for p_name in train_param_embed_event:
                if p_name in n:
                    p.requires_grad = True
                    train_param_embed_event_list.append(n)

            for p_name in train_param_vit_rgb:
                if p_name in n:
                    p.requires_grad = True
                    train_param_vit_rgb_list.append(n)

            for p_name in train_param_vit_event:
                if p_name in n:
                    p.requires_grad = True
                    train_param_vit_event_list.append(n)

            for p_name in train_param_vit_cross:
                if p_name in n:
                    p.requires_grad = True
                    train_param_vit_cross_list.append(n)

        print("params_embed_trained:", train_param_embed_rgb_list + train_param_embed_event_list)
        print("params_backbone_trained:", train_param_norm_list + train_param_vit_rgb_list + train_param_vit_event_list + train_param_vit_cross_list)

        #  == For FE108 ==
        # param_dicts = [
        #     {"params": [p for n, p in net.named_parameters() if n in train_param_norm_list], "lr": 0.000001},
        #     {"params": [p for n, p in net.named_parameters() if n in train_param_embed_rgb_list], "lr": 0.00001},
        #     {"params": [p for n, p in net.named_parameters() if n in train_param_embed_event_list], "lr": 0.00005},
        #     {"params": [p for n, p in net.named_parameters() if n in train_param_vit_rgb_list], "lr": 0.00002},
        #     {"params": [p for n, p in net.named_parameters() if n in train_param_vit_event_list], "lr": 0.0001},
        #     {"params": [p for n, p in net.named_parameters() if n in train_param_vit_cross_list], "lr": 0.00005},
        # ]

        #  == For Other Datasets ==
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if n in train_param_norm_list], "lr": 0.000001},
            {"params": [p for n, p in net.named_parameters() if n in train_param_embed_rgb_list], "lr": 0.00005},
            {"params": [p for n, p in net.named_parameters() if n in train_param_embed_event_list], "lr": 0.00005},
            {"params": [p for n, p in net.named_parameters() if n in train_param_vit_rgb_list], "lr": 0.0001},
            {"params": [p for n, p in net.named_parameters() if n in train_param_vit_event_list], "lr": 0.0001},
            {"params": [p for n, p in net.named_parameters() if n in train_param_vit_cross_list], "lr": 0.00005},
        ]
    #
    # ===== LoRA Tuning =====
    # for n, p in net.named_parameters():
    #     p.requires_grad = False
    #
    # train_param = ["lora"]
    # train_param_list = []
    #
    # if 'vipt' in train_type:
    #     for n, p in net.named_parameters():
    #         for p_name in train_param:
    #             if p_name in n:
    #                 p.requires_grad = True
    #                 train_param_list.append(n)
    #
    #     print("params_trained:", train_param_list)
    #
    #     #  == For Other Datasets ==
    #     param_dicts = [
    #         {"params": [p for n, p in net.named_parameters() if n in train_param_list], "lr": 0.001},
    #     ]

    # ===== Fine-tune =====
    for n, p in net.named_parameters():
        p.requires_grad = False

    train_param_norm = ["backbone.norm_rgb.","backbone.norm_event."]
    train_param_embed_rgb =  ["backbone.patch_embed_rgb."]
    train_param_vit_rgb = ["backbone.blocks_rgb." + str(i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    train_param_embed_event =  ["backbone.patch_embed_event."]
    train_param_vit_event = ["backbone.blocks_event." + str(i) for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    train_param_vit_cross = ["backbone.blocks_cross." + str(i) for i in [0,1,2,3]]

    train_param_norm_list = []
    train_param_embed_rgb_list = []
    train_param_embed_event_list = []
    train_param_vit_rgb_list = []
    train_param_vit_event_list = []
    train_param_vit_cross_list = []

    if 'vipt' in train_type:
        for n, p in net.named_parameters():
            for p_name in train_param_norm:
                if p_name in n:
                    p.requires_grad = True
                    train_param_norm_list.append(n)

            for p_name in train_param_embed_rgb:
                if p_name in n:
                    p.requires_grad = True
                    train_param_embed_rgb_list.append(n)

            for p_name in train_param_embed_event:
                if p_name in n:
                    p.requires_grad = True
                    train_param_embed_event_list.append(n)

            for p_name in train_param_vit_rgb:
                if p_name in n:
                    p.requires_grad = True
                    train_param_vit_rgb_list.append(n)

            for p_name in train_param_vit_event:
                if p_name in n:
                    p.requires_grad = True
                    train_param_vit_event_list.append(n)

            for p_name in train_param_vit_cross:
                if p_name in n:
                    p.requires_grad = True
                    train_param_vit_cross_list.append(n)

        print("params_embed_trained:", train_param_embed_rgb_list + train_param_embed_event_list)
        print("params_backbone_trained:", train_param_norm_list + train_param_vit_rgb_list + train_param_vit_event_list + train_param_vit_cross_list)

        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if n in train_param_norm_list], "lr": 0.000001},
            {"params": [p for n, p in net.named_parameters() if n in train_param_embed_rgb_list], "lr": 0.000001},
            {"params": [p for n, p in net.named_parameters() if n in train_param_embed_event_list], "lr": 0.000001},
            {"params": [p for n, p in net.named_parameters() if n in train_param_vit_rgb_list], "lr": 0.000001},
            {"params": [p for n, p in net.named_parameters() if n in train_param_vit_event_list], "lr": 0.000001},
            {"params": [p for n, p in net.named_parameters() if n in train_param_vit_cross_list], "lr": 0.000001},
        ]

    # =========================================================
    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        # lr_scheduler = None
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
