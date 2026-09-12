import logging

from models.networks import architectures as M

logger = logging.getLogger("base")

def define_network(opt):
    opt_net = opt["network"]
    which_model = opt_net["which_model"]
    setting = opt_net["setting"]
    netG = getattr(M, which_model)(**setting)                
    return netG
