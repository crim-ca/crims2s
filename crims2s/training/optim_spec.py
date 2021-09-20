"""Defines a domain specific language where we can configure an optimization setup
(optimizer + scheduler) from a dictionary. Ex::

optim_spec = {
    '_target_': 'torch.optim.Adam',
    'lr': 1e-3,
    'params': [
        {'_regex_': 'lin1(.*)$', 'lr': 1e-4},
        {'_regex_': 'lin2'},
        {'_regex_': '.*'}
    ],
    'scheduler': {
        '_target_': 'torch.optim.lr_scheduler.ReduceLROnPlateau',
        'factor': 1e-6
    }
}

optimizers, schedulers = create_optim(m, optim_spec)"""

import pydoc
import re

# optim spec:  optimizer_name, list of (regex, params) pairs, params, scheduler


def create_optim(model, optim_spec):
    """Create an optimization setup from a specification. 
    
    Returns:
        optimizers: The list of specified optimizers.
        schedulers: The list of specified schedulers."""
    if not isinstance(optim_spec, list):
        optim_spec = [optim_spec]

    optimizers = []
    schedulers = []
    for spec in optim_spec:
        optimizer, scheduler = make_one_optimizer(model, **spec)
        optimizers.append(optimizer)

        if scheduler is not None:
            schedulers.append(scheduler)

    unassigned_params = find_unassigned_params(model, optimizers)
    if unassigned_params:
        raise RuntimeError(f"There are unassigned params: {unassigned_params}")

    return optimizers, schedulers


def find_unassigned_params(model, optimizers):
    assigned_params = set()
    for opt in optimizers:
        for group in opt.param_groups:
            params = group["params"]
            for p in params:
                assigned_params.add(id(p))

    id_to_name = {id(p): name for name, p in model.named_parameters()}
    all_params = set(id_to_name)

    unassigned = all_params - assigned_params

    return set([id_to_name[i] for i in unassigned])


def make_one_optimizer(model, _target_=None, scheduler=None, params=None, **kwargs):
    params = resolve_param_spec(model, params)

    optimizer_cls = pydoc.locate(_target_)
    optimizer = optimizer_cls(params, **kwargs)

    if scheduler is not None:
        scheduler = resolve_scheduler_spec(optimizer, **scheduler)
    else:
        scheduler = None

    return optimizer, scheduler


def resolve_scheduler_spec(optimizer, _target_=None, _monitor_=None, **kwargs):
    scheduler_cls = pydoc.locate(_target_)
    scheduler = scheduler_cls(optimizer, **kwargs)

    if _monitor_:
        # If there is a _monitor_ key, output a scheduler spec in the pytorch lightning format.
        return {"scheduler": scheduler, "monitor": _monitor_}
    else:
        return scheduler


def resolve_param_spec(model, param_spec):
    param_spec = [dict(s) for s in param_spec]

    regexes = [spec.pop("_regex_") for spec in param_spec]
    param_lists = make_param_lists(model, regexes)

    # In each param list, replace the param regex with the param list.
    for param_list, group in zip(param_lists, param_spec):
        group["params"] = param_list

    return param_spec


def make_param_lists(model, regexes):
    all_params = {name: param for name, param in model.named_parameters()}

    param_lists = []
    assigned_param_names = set({})
    for regex_str in regexes:
        param_list = []
        regex = re.compile(regex_str)
        for k in all_params:
            if regex.search(k):
                param_list.append(all_params[k])
                assigned_param_names.add(k)

        if len(param_list) == 0:
            raise RuntimeError(
                f"Regex {regex_str} resulted in an empty parameters list."
            )

        param_lists.append(param_list)

        # Update all params dict to make sure nothing is assigned twice.
        all_params = {
            k: v for k, v in all_params.items() if k not in assigned_param_names
        }

    return param_lists
