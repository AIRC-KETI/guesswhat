import copy
from platform import architecture

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import (
    AutoConfig,
    CLIPConfig,
    VisionTextDualEncoderConfig,
)

from models.fusion.configuration_fusion import FusionConfig

logger = logging.get_logger(__name__)


class OldQuestionerConfig(PretrainedConfig):
    model_type: str = "questioner"
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


class QuestionerConfig(PretrainedConfig):
    r"""
    [`QuestionerConfig`] is the configuration class to store the configuration of a
    [`QuestionerModel`]. It is used to instantiate [`QuestionerModel`] model according to the
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

    model_type = "questioner"
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
        """
        vision_text_model_config: {
            'return_dict': True, 'output_hidden_states': False, 
            output_attentions': False, 'torchscript': False, 'torch_dtype': torch.float32,
            ' use_bfloat16': False, 'tf_legacy_loss': False, 'pruned_heads': {},
            'tie_word_emb eddings': True, 'is_encoder_decoder': False, 'is_decoder': False,
            'cross_attenti on_hidden_size': None, 'add_cross_attention': False, 'tie_encoder_decoder': False,
            'max_length': 20, 'min_length': 0, 'do_sample': False, 'early_stopping': False, 'num_beams': 1,
            'num_beam_groups': 1, 'diversity_penalty': 0.0, 'temperature' : 1.0, 'top_k': 50, 'top_p': 1.0, 'typical_p': 1.0,
            'repetition_penalty': 1.0, 'length_penalty': 1.0, 'no_repeat_ngram_size': 0, 'encoder_no_repeat_ngram_size': 0,
            'bad_words_ids': None, 'num_return_sequences': 1, 'chunk_size_feed_forward': 0, 'output_scores': False,
            'return_dict_in_generate': False, 'forced_bos_token_ id': None, 'forced_eos_token_id': None,
            'remove_invalid_values': False, 'exponen tial_decay_length_penalty': None, 'suppress_tokens': None,
            'begin_suppress_token s': None, 'architectures': ['CLIPModel'], 'finetuning_task': None, 'id2label': { 0: 'LABEL_0', 1: 'LABEL_1'}, 'label2id': {'LABEL_0': 0, 'LABEL_1': 1}, 'tokenize r_class': None, 'prefix': None, 'bos_token_id': None, 'pad_token_id': None, 'eos _token_id': None, 'sep_token_id': None, 'decoder_start_token_id': None, 'task_sp ecific_params': None, 'problem_type': None, '_name_or_path': '', '_commit_hash': '57c216476eefef5ab752ec549e440a49ae4ae5f3', 'transformers_version': None, 'init ializer_factor': 1.0, 'text_config': {'return_dict': True, 'output_hidden_states ': False, 'output_attentions': False, 'torchscript': False, 'torch_dtype': None, 'use_bfloat16': False, 'tf_legacy_loss': False, 'pruned_heads': {}, 'tie_word_e mbeddings': True, 'is_encoder_decoder': False, 'is_decoder': False, 'cross_atten tion_hidden_size': None, 'add_cross_attention': False, 'tie_encoder_decoder': Fa lse, 'max_length': 20, 'min_length': 0, 'do_sample': False, 'early_stopping': Fa lse, 'num_beams': 1, 'num_beam_groups': 1, 'diversity_penalty': 0.0, 'temperatur e': 1.0, 'top_k': 50, 'top_p': 1.0, 'typical_p': 1.0, 'repetition_penalty': 1.0, 'length_penalty': 1.0, 'no_repeat_ngram_size': 0, 'encoder_no_repeat_ngram_size ': 0, 'bad_words_ids': None, 'num_return_sequences': 1, 'chunk_size_feed_forward ': 0, 'output_scores': False, 'return_dict_in_generate': False, 'forced_bos_toke n_id': None, 'forced_eos_token_id': None, 'remove_invalid_values': False, 'expon ential_decay_length_penalty': None, 'suppress_tokens': None, 'begin_suppress_tok ens': None, 'architectures': None, 'finetuning_task': None, 'id2label': {0: 'LAB EL_0', 1: 'LABEL_1'}, 'label2id': {'LABEL_0': 0, 'LABEL_1': 1}, 'tokenizer_class ': None, 'prefix': None, 'bos_token_id': 0, 'pad_token_id': 1, 'eos_token_id': 2 , 'sep_token_id': None, 'decoder_start_token_id': None, 'task_specific_params': None, 'problem_type': None, '_name_or_path': '', 'transformers_version': '4.26.0 .dev0', 'vocab_size': 49408, 'hidden_size': 512, 'intermediate_size': 2048, 'pro jection_dim': 512, 'dropout': 0.0, 'num_hidden_layers': 12, 'num_attention_heads ': 8, 'max_position_embeddings': 77, 'layer_norm_eps': 1e-05, 'hidden_act': 'qui ck_gelu', 'initializer_range': 0.02, 'initializer_factor': 1.0, 'attention_dropo ut': 0.0, 'model_type': 'clip_text_model'}, 'text_config_dict': None, 'vision_co nfig': {'return_dict': True, 'output_hidden_states': False, 'output_attentions': False, 'torchscript': False, 'torch_dtype': None, 'use_bfloat16': False, 'tf_le gacy_loss': False, 'pruned_heads': {}, 'tie_word_embeddings': True, 'is_encoder_ decoder': False, 'is_decoder': False, 'cross_attention_hidden_size': None, 'add_ cross_attention': False, 'tie_encoder_decoder': False, 'max_length': 20, 'min_le ngth': 0, 'do_sample': False, 'early_stopping': False, 'num_beams': 1, 'num_beam _groups': 1, 'diversity_penalty': 0.0, 'temperature': 1.0, 'top_k': 50, 'top_p'$
        """
        architectures = vision_text_model_config["architectures"]
        if 'CLIPModel' in architectures[0]:
            vision_text_model_type = 'clip'
        else:
            pass
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