from typing import Dict, Type

# Action registry
_ACTION_REGISTRY: Dict[str, Type] = {}

class ActionMeta(type):
    def __new__(cls, name, bases, dct):
        klass = super().__new__(cls, name, bases, dct)
        # Register the action if it has an 'action' attribute and it's not "base"
        if hasattr(klass, 'action') and isinstance(getattr(klass, 'action'), str) and klass.action != "base":
            _ACTION_REGISTRY[getattr(klass, 'action')] = klass
        return klass

class BaseAction(metaclass=ActionMeta):
    action: str = "base"
    def __init__(self, *args, **kwargs):
        pass
    def step(self, context, synced_data: dict):
        raise NotImplementedError
    def is_completed(self) -> bool:
        raise NotImplementedError
    def failed(self) -> bool:
        return False

def parse_action(d: dict) -> BaseAction:
    if "action" not in d:
        raise ValueError("Missing action field in action dictionary")
    
    action_name = d["action"]
    if action_name not in _ACTION_REGISTRY:
        # Give a helpful error message about which actions ARE registered
        available_actions = list(_ACTION_REGISTRY.keys())
        raise ValueError(f"Unsupported action: '{action_name}'. Registered actions: {available_actions}")
    
    klass = _ACTION_REGISTRY[action_name]
    # Pass all other keys from the dict as constructor arguments
    kwargs = {k: v for k, v in d.items() if k != "action"}
    return klass(**kwargs)
