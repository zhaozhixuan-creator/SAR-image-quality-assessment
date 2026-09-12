def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for a module or a list of modules."""
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
