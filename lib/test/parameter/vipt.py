import os
from lib.test.utils import TrackerParams
from lib.config.vipt.config import cfg, update_config_from_file
from lib.test.evaluation.local import EnvironmentSettings


def parameters(yaml_name:str,experiment_id,epoch):
    params = TrackerParams()
    project_dir = EnvironmentSettings().project_dir
    checkpoint_dir = EnvironmentSettings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(project_dir, 'experiments/vipt/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(checkpoint_dir, experiment_id+"/checkpoints/train/vipt/deep_rgbe/SIAMTrack_ep%04d.pth.tar" % (epoch))
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
