# from .environment import env_settings
from .local import EnvironmentSettings

class Settings:
    """ Training settings, e.g. the paths to datasets and networks."""
    def __init__(self):
        self.set_default()

    def set_default(self):
        # self.env = env_settings()
        self.env = EnvironmentSettings()
        self.use_gpu = True


