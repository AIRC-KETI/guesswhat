from typing import Union
import os
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import BertConfig
logger = logging.get_logger(__name__)


class FusionConfig(PretrainedConfig):
    r"""
    [`FusionConfig`] is the configuration class to store the configuration of a
    [`FusionModel`]. It is used to instantiate [`FusionModel`] model according to the
    specified arguments, defining the text model, vision model, and fusion model configs.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
    """

    model_type = "fusion"


    def __init__(
        self,
        hidden_act = "quick_gelu",
        hidden_size = 512,
        initializer_range = 0.02,
        intermediate_size = 2048,
        layer_norm_eps = 1e-12,
        max_position_embeddings = 77,
        model_type = "fusion",
        num_attention_heads = 8,
        num_hidden_layers = 12,
        use_cache = True,
        dropout = 0.0,
        attention_dropout = 0.0,
        initializer_factor = 1.0,
        summary_type = 'mean',
        **kwargs
    ):
        super().__init__(fusion_model_type=None, **kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_factor = initializer_factor
        self.summary_type = summary_type,

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        return FusionConfig()