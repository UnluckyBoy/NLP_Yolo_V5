import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                # self.update(yaml.load(fo.read()))
                """Yaml 5.1版本之后就修改了需要指定Loader，通过默认加载 器（FullLoader）禁止执行任意函数，以下三种方式可用"""
                self.update(yaml.safe_load(fo.read()))
                # self.update(yaml.load(fo.read(), Loader=yaml.FullLoader))
                # self.update(yaml.load(fo.read(), Loader=yaml.CLoader))

        super(YamlParser, self).__init__(cfg_dict)

    
    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.safe_load(fo.read()))

    
    def merge_from_dict(self, config_dict):
        self.update(config_dict)


def get_config(config_file=None):
    return YamlParser(config_file=config_file)


if __name__ == "__main__":
    cfg = YamlParser(config_file="../configs/Yolo_v5.yaml")
    cfg.merge_from_file("../configs/deep_sort.yaml")

    import ipdb; ipdb.set_trace()