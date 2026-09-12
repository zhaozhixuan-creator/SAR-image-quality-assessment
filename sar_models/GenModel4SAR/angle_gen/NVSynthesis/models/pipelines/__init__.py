import logging
import importlib

logger = logging.getLogger("base")


def create_pipl(opt, device=None):
    pipeline = opt["pipeline"]
    subpipeline = opt["sub_pipeline"]
    M = importlib.import_module("models.pipelines.%s.%s.pipL_utils" % (pipeline, subpipeline))

    m = M.DenoisingPiPL(opt, device)
    assert m is not None, "create pipL FAILED: from models.pipelines.%s.%s.pipL_utils import DenoisingPiPL" % (pipeline, subpipeline)
    logger.info("PiPL [{:s}] is created.".format(m.__class__.__name__))
    return m