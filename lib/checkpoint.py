from __future__ import annotations

import inspect

import torch as th


def load_state_dict(path, map_location):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(th.load).parameters:
        kwargs["weights_only"] = True
    return th.load(path, **kwargs)

