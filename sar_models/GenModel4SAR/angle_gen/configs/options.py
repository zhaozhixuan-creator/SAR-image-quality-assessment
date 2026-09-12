import os
import os.path as osp

import yaml

from NVSynthesis.utils import OrderedYaml, get_timestamp


Loader, Dumper = OrderedYaml()

def parse(opt_path, is_train=True):
    def resolve_repo_path(path):
        if not isinstance(path, str):
            return path
        path = osp.expanduser(path)
        if osp.isabs(path):
            return path
        return osp.abspath(osp.join(opt["path"]["root"], path))

    def get_basic_setup():
        # get experiment name with default "ohne"
        opt["name"] = opt.get("name", "ohne")

        # get basic root dirs
        if "path" not in opt:
            opt["path"] = {}
        opt["path"]["root"] = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
        opt["path"]["config_root"] = osp.join(opt["path"]["root"], "configs")
    
    def setting_args():
        gpu_list = ",".join(str(x) for x in opt["gpu_ids"])
        opt["gpu_ids"] = list(range(len(opt["gpu_ids"])))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
        opt["is_train"] = is_train

        ## schedule
        tmp_config_path = osp.join(
                        opt["path"]["config_root"], "_base_", "setting",
                        opt["setting"]["base"] + ".yml"
                    )
        # ./configs/_base_/setting/train.yml or test.yml
        with open(tmp_config_path, mode="r") as f:
            opt_tmp = yaml.load(f, Loader=Loader)
            opt.update(opt_tmp)
        # change some options for debug mode
        if "debug" in opt["name"]:
            opt["setting"]["val_freq"] = 1
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 1000
    
    def network_args():
        network_base = opt["network"]["base"]
        tmp_config_path = osp.join(
                        opt["path"]["config_root"], "_base_", "model",
                        network_base + ".yml"
                    )
        # ./configs/_base_/model/xunet.yml
        with open(tmp_config_path, mode="r") as f:
            opt_tmp = yaml.load(f, Loader=Loader)
            opt.update(opt_tmp) # ["network"]
    
    def data_args():
        tmp_config_path = osp.join(
                opt["path"]["config_root"], "_base_", "datasets", opt["datasets"]["base"] + ".yml"
            )
        # ./configs/_base_/datasets/car-default.yml
        with open(tmp_config_path, mode="r") as f:
            opt_dataset = yaml.load(f, Loader=Loader)
        
        for phase, dataset_list in opt_dataset["datasets"].items():
            if phase == "image_size":
                opt["datasets"]["image_size"] = dataset_list
                opt["network"]["setting"]["inputH"] = opt["datasets"]["image_size"]
                opt["network"]["setting"]["inputW"] = opt["datasets"]["image_size"]
                continue
            for name, dataset_opt in dataset_list.items():
                dataset_opt["name"] = name # 
                dataset_opt["phase"] = phase

                #   datasource: 
                tmp_config_path = osp.join(
                                opt["path"]["config_root"], "_base_", "datasets", "datasource",
                                dataset_opt["datasource"]["type"].split("/")[0],
                                dataset_opt["datasource"]["type"].split("/")[1] + ".yml"
                            )
                with open(tmp_config_path, mode="r") as f:
                    opt_tmp = yaml.load(f, Loader=Loader)
                    dataset_opt.update(opt_tmp["datasource"][phase])
                if "dataroot" in dataset_opt:
                    dataset_opt["dataroot"] = resolve_repo_path(dataset_opt["dataroot"])

                #   dataloader:
                dataset_opt["mode"] = dataset_opt["dataloader"]["mode"]
                dataset_opt["image_size"] = opt["datasets"]["image_size"]
                tmp_config_path = osp.join(
                                opt["path"]["config_root"], "_base_", "datasets", "dataloader",
                                dataset_opt["dataloader"]["type"] + ".yml"
                            )
                with open(tmp_config_path, mode="r") as f:
                    opt_tmp = yaml.load(f, Loader=Loader)
                    dataset_opt.update(opt_tmp["dataloader"][phase])
                if is_train:
                    dataset_opt["n_workers"] = dataset_opt["n_workers"] * len(opt["gpu_ids"])
                else:
                    dataset_opt["n_workers"] = 0
                    
            opt["datasets"][phase] = dataset_list

    def setup_dir():
        if is_train:
            experiments_root = os.path.join(
                                            opt["path"]["root"], 
                                            "experiments", 
                                            opt["name"]
                                           )
            if opt["pretrain"]:
                experiments_root = os.path.join(
                                                experiments_root, 
                                                opt["pretrain"]
                                            )
                if "pretrain_model_pth" not in opt["path"]:
                    opt["path"]["pretrain_model_pth"] = os.path.join(experiments_root, "models", "latest_adapted.pt")
            else:
                experiments_root = os.path.join(
                                                experiments_root, 
                                                get_timestamp()
                                            )
            opt["path"]["experiments_root"] = experiments_root
            opt["path"]["models"] = osp.join(experiments_root, "models")
            opt["path"]["training_state"] = osp.join(experiments_root, "training_state")
            opt["path"]["log"] = experiments_root
            opt["path"]["sde_state"] = osp.join(experiments_root, "sde_state")
        else:  # test
            experiments_root = os.path.join(
                                            opt["path"]["root"], 
                                            "experiments", 
                                            opt["name"]
                                           )
            experiments_root = os.path.join(
                                            experiments_root, 
                                            opt["pretrain"]
                                        )
            if "pretrain_model_pth" not in opt["path"]:
                opt["path"]["pretrain_model_pth"] = os.path.join(experiments_root, "models", "latest_adapted.pt") # To DO: add more flexibility here
                
            results_root = osp.join(opt["path"]["root"], "results")
            opt["path"]["results_root"] = osp.join(results_root, opt["name"], opt["pretrain"], get_timestamp())
            opt["path"]["log"] = opt["path"]["results_root"]    
        if opt["path"].get("pretrain_model_pth"):
            opt["path"]["pretrain_model_pth"] = resolve_repo_path(opt["path"]["pretrain_model_pth"])
    
    with open(opt_path, mode="r") as f:
        opt = yaml.load(f, Loader=Loader)

    ## get experiment name and some basic root paths
    get_basic_setup()
    ## train related
    setting_args()
    ## get parameters for network
    network_args()
    ## get parameters for dataset
    data_args()
    # path
    setup_dir()
    # convert to NoneDict, which return None for missing key.
    opt = dict_to_nonedict(opt)

    return opt


def dict2str(opt, indent_l=1):
    """dict to string for logger"""
    msg = ""
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_l * 2) + k + ":[\n"
            msg += dict2str(v, indent_l + 1)
            msg += " " * (indent_l * 2) + "]\n"
        else:
            msg += " " * (indent_l * 2) + k + ": " + str(v) + "\n"
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    """ convert to NoneDict, which return None for missing key.
    """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
