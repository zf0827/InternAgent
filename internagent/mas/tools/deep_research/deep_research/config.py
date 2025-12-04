import os
import warnings

DEFAULT_PROVIDER_PRIORITY = ["deepseek", "openai", "claude", "basic", "azure_openai"]


def get_api_config():
    return {
        "openai": {"config_list": [{"model": os.environ.get("OPENAI_MODEL", "gpt-4o"), "api_key": os.environ.get("OPENAI_API_KEY"), "base_url": os.environ.get("OPENAI_BASE_URL")}]},
        "claude": {"config_list": [{"model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), "api_key": os.environ.get("ANTHROPIC_API_KEY"), "base_url": os.environ.get("ANTHROPIC_BASE_URL")}]},
        "deepseek": {"config_list": [{"model": os.environ.get("DEEPSEEK_MODEL", "DeepSeek-V3.2"), "api_key": os.environ.get("DEEPSEEK_API_KEY"), "base_url": os.environ.get("DEEPSEEK_BASE_URL")}]},
        "basic": {"config_list": [{"model": os.environ.get("OPENAI_MODEL", "gpt-4o"), "api_key": os.environ.get("OPENAI_API_KEY"), "base_url": os.environ.get("OPENAI_BASE_URL")}]},
        "azure_openai": {"config_list": [{"model": os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o"), "api_key": os.environ.get("AZURE_OPENAI_API_KEY"), "base_url": os.environ.get("AZURE_OPENAI_ENDPOINT"), "api_type": "azure", "api_version": os.environ.get("AZURE_OPENAI_API_VERSION")}]},
    }


service_config = {"summary": get_api_config()["basic"], "deepsearch": get_api_config()["basic"], "code_explore": get_api_config()["basic"]}


def get_default_provider():
    return os.environ.get("DEFAULT_API_PROVIDER", DEFAULT_PROVIDER_PRIORITY[0])


def get_provider_by_priority():
    default_provider = get_default_provider()
    api_configs = get_api_config()
    if default_provider in api_configs:
        config = api_configs[default_provider]
        config_list = config.get("config_list", [])
        if config_list:
            api_key = config_list[0].get("api_key")
            if api_key and api_key.strip():
                return default_provider, config
    for provider in DEFAULT_PROVIDER_PRIORITY:
        if provider == default_provider:
            continue
        if provider in api_configs:
            config = api_configs[provider]
            config_list = config.get("config_list", [])
            if config_list:
                api_key = config_list[0].get("api_key")
                if api_key and api_key.strip():
                    return provider, config
    raise ValueError("No valid API provider found. Please configure at least one API key.")


def validate_and_get_fallback_config(api_type: str = None, service_type: str = ""):
    if api_type is None:
        provider, config = get_provider_by_priority()
        return provider, config
    if api_type not in get_api_config():
        raise ValueError(f"API type '{api_type}' not found in configuration")
    api_config = get_api_config()[api_type]
    if service_type and service_type in service_config:
        api_config = service_config[service_type]
        config_source = f"service:{service_type}"
    else:
        config_source = api_type
    config_list = api_config.get("config_list", [])
    if not config_list:
        raise ValueError(f"No config_list found for API type '{api_type}'")
    primary_config = config_list[0]
    primary_api_key = primary_config.get("api_key")
    if not primary_api_key or primary_api_key.strip() == "":
        warnings.warn(f"API key for '{config_source}' is None or empty. Using provider priority fallback...")
        provider, config = get_provider_by_priority()
        return provider, config
    else:
        return config_source, api_config


def get_llm_config(api_type: str = None, timeout: int = 240, temperature: float = 0.1, top_p=0.95, service_type: str = "", validate_api_key: bool = True):
    if validate_api_key or api_type is None:
        _, api_config = validate_and_get_fallback_config(api_type, service_type)
        api_config = api_config.copy()
    else:
        api_config = get_api_config()[api_type]
        if service_type and service_type in service_config:
            api_config = service_config[service_type]
    api_config["timeout"] = timeout
    api_config["temperature"] = temperature
    api_config["top_p"] = top_p
    return api_config

