from dynaconf import Dynaconf

settings = Dynaconf(
    environments=True,
    envvar_prefix="SMART_COUNTER",
    merge_enabled=True,
    settings_file=['settings.yaml'],
    root_path=__file__
)
