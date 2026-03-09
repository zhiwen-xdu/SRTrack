class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/czwos/Project/SIAMTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/czwos/Project/SIAMTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/czwos/Project/SIAMTrack/pretrained'
        self.fe108_dir = '/home/czwos/Data/FE108'
        self.visevent_dir = '/home/czwos/Data/VisEvent'
        self.coesot_dir = '/home/czwos/Data/COESOT_V2'
        self.depthtrack_dir = '/home/czwos/Data/DepthTrack'
        self.lasher_dir = '/home/czwos/Data/LasHeR'
