import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import (
    AutoConfig,
    VisionTextDualEncoderConfig,
)

from models.fusion.configuration_fusion import FusionConfig

logger = logging.get_logger(__name__)


class OldOracleConfig(PretrainedConfig):
    model_type: str = "oracle"
    is_composition: bool = True

    def __init__(self, clip_config=None, fusion_config=None, **kwargs):
        super().__init__(**kwargs)
        
        clip_config_dict = kwargs.pop("clip_config", None)
        fusion_config_dict = kwargs.pop("fusion_config", None)
        
        if clip_config_dict is not None:
            clip_config = clip_config_dict
        if clip_config is None:
            clip_config = {}
            logger.info("clip_config is None. Initializing the VisionTextDualEncoderConfig with default values.")
        if fusion_config_dict is not None:
            fusion_config = fusion_config_dict
        if fusion_config is None:
            fusion_config = {}
            logger.info("fusion_config is None. Initializing the FusionConfig with default values.")
        
        self.clip_config = VisionTextDualEncoderConfig(**clip_config)
        self.fusion_config = FusionConfig(**fusion_config)

    @property
    def get_fusion_config(self):
        return self.fusion_config
    @property
    def get_clip_config(self):
        return self.clip_config


class OracleConfig(PretrainedConfig):
    r"""
    [`OracleConfig`] is the configuration class to store the configuration of a
    [`OracleModel`]. It is used to instantiate [`OracleModel`] model according to the
    specified arguments, defining the text model, vision model, and fusion model configs.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vision_language_model_config (`dict`):
            Dictionary of configuration options that defines text model and vision model config.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "Oracle"
    is_composition = True

    def __init__(self, projection_dim=None, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(**kwargs)

        if "vision_text_model_config" not in kwargs:
            raise ValueError("`vision_text_model_config` can not be `None`.")
        
        if "fusion_model_config" not in kwargs:
            raise ValueError("`fusion_model_config` can not be `None`.")

        vision_text_model_config = kwargs.pop("vision_text_model_config")
        fusion_model_config = kwargs.pop("fusion_model_config")

        vision_text_model_type = vision_text_model_config.pop("model_type")
        fusion_model_type = fusion_model_config.pop("model_type")

        # set projection_dim for vtm and fusion model
        if projection_dim is not None:
            vision_text_model_config["projection_dim"] = projection_dim
        else:
            projection_dim = vision_text_model_config["projection_dim"]
        fusion_model_config["hidden_size"] = projection_dim

        self.vision_text_model_config = AutoConfig.for_model(vision_text_model_type, **vision_text_model_config)  # VisionTextDualEncoderConfig.from_pretrained(vision_text_model_config)
        self.fusion_model_config = FusionConfig.for_model(fusion_model_type, **fusion_model_config)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_vision_text_fusion_configs(cls, vision_text_model_config: PretrainedConfig, fusion_model_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a [`VisionTextDualEncoderConfig`] (or a derived class) from text model configuration and vision
        model configuration.
        Returns:
            [`VisionTextDualEncoderConfig`]: An instance of a configuration object
        """
        return cls(vision_text_model_config=vision_text_model_config.to_dict(), fusion_model_config=fusion_model_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_text_model_config"] = self.vision_text_model_config.to_dict()
        output["fusion_model_config"] = self.fusion_model_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output