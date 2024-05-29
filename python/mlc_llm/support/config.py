"""
A common base class for configuration. A configuration could be initialized from its constructor,
a JSON string or a JSON file, and irrelevant fields during initialization are automatically moved
to the `kwargs` field.

Take model configuration as an example: it is usually a JSON file in HuggingFace that contains
the model's hyperparameters. For instance, Vicuna-13b-v1.5-16k contains the following
[JSON file](https://huggingface.co/lmsys/vicuna-13b-v1.5-16k/blob/main/config.json).
The base class allows us to load the configuration from this JSON file, moving irrelevant fields
into `kwargs`, such as `transformers_version` and `use_cache`.
"""

# pylint: disable=too-few-public-methods
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from . import logging
from .style import bold, red

logger = logging.getLogger(__name__)

ConfigClass = TypeVar("ConfigClass", bound="ConfigBase")


@dataclasses.dataclass
class ConfigBase:
    """Base class for configurations, providing a common interface for loading configs from a
    JSON file or a dict. It requires the subclasses to be dataclasses, and has an `kwargs` field
    that stores the extra fields that are not defined in the dataclass.
    """

    @classmethod
    def from_dict(cls: Type[ConfigClass], source: Dict[str, Any]) -> ConfigClass:
        """Create a config object from a dictionary.

        Parameters
        ----------
        source : Dict[str, Any]
            Source to create config from, usually loaded from `config.json` in HuggingFace style.

        Returns
        -------
        cfg : ConfigClass
            An instance of the config object.
        """
        field_names = [field.name for field in dataclasses.fields(cls)]  # type: ignore[arg-type]
        fields = {k: v for k, v in source.items() if k in field_names}
        kwargs = {k: v for k, v in source.items() if k not in field_names}
        return cls(**fields, kwargs=kwargs)  # type: ignore[call-arg]

    @classmethod
    def from_file(cls: Type[ConfigClass], source: Path) -> ConfigClass:
        """Create a config object from a file.

        Parameters
        ----------
        cfg_cls : Type[ConfigClass]
            The config class to create, for example, LlamaConfig.

        source : pathlib.Path
            Path to the source file, usually `config.json` in HuggingFace repo.

        Returns
        -------
        cfg : ConfigClass
            An instance of the config object.
        """
        with source.open("r", encoding="utf-8") as in_file:
            return cls.from_dict(json.load(in_file))

    def asdict(self):
        """Convert the config object to a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the config object.
        """
        result = dataclasses.asdict(self)
        result.pop("kwargs")
        return result


class ConfigOverrideBase:
    """Base class for ConfigOverride, providing a common interface for overriding configs.
    It requires the subclasses to be dataclasses.
    """

    def apply(self, config):
        """Apply the overrides to the given config."""
        updated = config.asdict()
        for field in dataclasses.fields(self):
            key = field.name
            value = getattr(self, key)
            if value is None:
                continue
            if key not in updated:
                logger.warning(
                    "%s: Cannot override %s, because %s does not have this field",
                    red("Warning"),
                    bold(key),
                    bold(type(config).__name__),
                )
            else:
                logger.info(  # pylint: disable=logging-fstring-interpolation
                    f"Overriding {bold(key)} from {updated[key]} to {value}"
                )
                updated[key] = value
        return type(config).from_dict(updated)


__all__ = ["ConfigBase", "ConfigOverrideBase"]
