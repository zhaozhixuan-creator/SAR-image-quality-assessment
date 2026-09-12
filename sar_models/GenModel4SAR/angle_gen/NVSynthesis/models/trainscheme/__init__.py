import logging
import importlib

logger = logging.getLogger("base")


def create_scheme(opt):
    scheme = opt["scheme"]
    subscheme = opt["sub_scheme"]
    # despeckle/models/trainscheme/supervised/base.py
    M = importlib.import_module("models.trainscheme.%s.%s" % (scheme, subscheme))

    m = M.CustomizedScheme(opt)
    assert m is not None, "create training scheme FAILED: from %s.%s import CustomizedScheme" % (scheme, subscheme)
    logger.info("Training Scheme [{:s}] is created.".format(m.__class__.__name__))
    return m