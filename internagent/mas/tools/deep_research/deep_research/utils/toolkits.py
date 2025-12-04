from typing import List, Callable
from functools import wraps
import importlib
import inspect


def stringify_output(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        return result

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper


def _import_register_function():
    try:
        mod = importlib.import_module("autogen.tools.function_utils")
        return getattr(mod, "register_function")
    except Exception:
        mod = importlib.import_module("autogen")
        return getattr(mod, "register_function")


def register_toolkits(config: List[Callable], caller, executor, **kwargs):
    register_function = _import_register_function()
    for tool in config:
        if isinstance(tool, type):
            register_tookits_from_cls(caller, executor, tool, **kwargs)
            continue
        tool_dict = {"function": tool} if callable(tool) else tool
        tool_function = tool_dict["function"]
        name = tool_dict.get("name", tool_function.__name__)
        description = tool_dict.get("description", tool_function.__doc__)
        register_function(stringify_output(tool_function), caller=caller, executor=executor, name=name, description=description)


def register_tookits_from_cls(caller, executor, cls: type, include_private: bool = False):
    if include_private:
        funcs = [func for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("__")]
    else:
        funcs = [func for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("__") and not func.startswith("_")]
    register_toolkits([getattr(cls, func) for func in funcs], caller, executor)
