from gc import freeze
from typing import Any, Optional, Tuple, Union
from xmlrpc.client import Boolean

from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderConfig,
    CLIPModel,
    CLIPConfig,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
)
from transformers.utils import (
    logging,
    ModelOutput,
)

import torch
import torch.nn as nn
from torch.nn import (
    MSELoss,
    CrossEntropyLoss,
    BCEWithLogitsLoss,
)

from models.fusion.fusion import FusionModel
from models.guesser.configuration_guesser import GuesserConfig


logger = logging.get_logger(__name__)
# -----------------------------------------------------------------
# CUSTOM CLIP -----------------------------------------------
# -----------------------------------------------------------------
class MyCLIPOutput(ModelOutput):
    """
    Args:
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`MyVisionTextDualEncoderOutput`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`MyVisionTextDualEncoderOutput`):
            The output of the [`CLIPVisionModel`].
    """

    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class MyCLIPModel(CLIPModel):
    base_model_prefix = "my_clip"
    def __init__(
        self,
        config: Optional[CLIPConfig] = None,
    ):
        super().__init__(config)
        self.freeze_clip()
    
    def freeze_clip(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MyCLIPOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        # [bsz, row_patches*col_patches, model_dim] [model_dim, model_dim2]
        # [bsz, model_dim] [model_dim, model_dim2]
        image_embeds = vision_outputs[0]
        image_embeds = self.visual_projection(image_embeds)

        # text_embeds = text_outputs[0]  # seq
        text_embeds = text_outputs[1]  # pooled
        text_embeds = self.text_projection(text_embeds).unsqueeze(1)

        if not return_dict:
            output = (text_embeds, image_embeds, text_outputs, vision_outputs)
            return output

        return MyCLIPOutput(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def get_image_seq_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        b, seq, features = vision_outputs.last_hidden_state.size()
        image_features = self.visual_projection(vision_outputs.last_hidden_state.view(b * seq, -1))

        return image_features.view(b, seq, -1)
    
    def get_text_seq_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        b, seq, features = text_outputs.last_hidden_state.size()
        text_features = self.text_projection(text_outputs.last_hidden_state.view(b * seq, -1))

        return text_features.view(b, seq, -1)

# -----------------------------------------------------------------
# CUSTOM TextDualEncoder -----------------------------------------------
# -----------------------------------------------------------------
class MyVisionTextDualEncoderOutput(ModelOutput):
    """
    Args:
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`CLIPTextModel`].
        image_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The image embeddings obtained by applying the projection layer to the pooled output of [`CLIPVisionModel`].
        text_model_output(`MyVisionTextDualEncoderOutput`):
            The output of the [`CLIPTextModel`].
        vision_model_output(`MyVisionTextDualEncoderOutput`):
            The output of the [`CLIPVisionModel`].
    """

    text_embeds: torch.FloatTensor = None
    image_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    vision_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k] if k not in ["text_model_output", "vision_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class MyVisionTextDualEncoderModel(VisionTextDualEncoderModel):
    base_model_prefix = "my_vision_text_dual_encoder"
    def __init__(
        self,
        config: Optional[VisionTextDualEncoderConfig] = None,
        vision_model: Optional[PreTrainedModel] = None,
        text_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config, vision_model, text_model)
        self.freeze_clip()
    
    def freeze_clip(self):
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MyVisionTextDualEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        # [bsz, row_patches*col_patches, model_dim] [model_dim, model_dim2]
        # [bsz, model_dim] [model_dim, model_dim2]
        image_embeds = vision_outputs[0]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[0]
        text_embeds = self.text_projection(text_embeds)

        if not return_dict:
            output = (text_embeds, image_embeds, text_outputs, vision_outputs)
            return output

        return MyVisionTextDualEncoderOutput(
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )

    def get_image_seq_features(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        b, seq, features = vision_outputs.last_hidden_state.size()
        image_features = self.visual_projection(vision_outputs.last_hidden_state.view(b * seq, -1))

        return image_features.view(b, seq, -1)
    
    def get_text_seq_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        b, seq, features = text_outputs.last_hidden_state.size()
        text_features = self.text_projection(text_outputs.last_hidden_state.view(b * seq, -1))

        return text_features.view(b, seq, -1)

# -----------------------------------------------------------------
# Guesser MODEL ----------------------------------------------------
# -----------------------------------------------------------------


class GuesserModel(PreTrainedModel):
    config_class = GuesserConfig
    base_model_prefix = "guesser"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, 
            config: GuesserConfig, 
            *inputs,
            vision_text_pretrained=None,
            vision_model_path=None, 
            language_model_path=None,
            **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if vision_text_pretrained is not None:
            self.vision_text_model = MyVisionTextDualEncoderModel.from_pretrained(vision_text_pretrained)
        elif language_model_path is not None and vision_model_path is not None:
            self.vision_text_model = MyVisionTextDualEncoderModel.from_vision_text_pretrained(vision_model_path, language_model_path)
        elif "clip" in config.vision_text_model_config.model_type:
            self.vision_text_model = MyCLIPModel(config.vision_text_model_config)
        else:
            self.vision_text_model = MyVisionTextDualEncoderModel(config.vision_text_model_config)
        self.fusion_model = FusionModel(config.fusion_model_config)
    
        vision_config = config.vision_text_model_config.vision_config
        image_size = vision_config.image_size
        patch_size = vision_config.patch_size
        self.image_seq_len = (image_size//patch_size)
        self.image_special_tokens_left = 1
    
    def _build_our_attention_mask(self, input_ids, attention_mask, bbox, name='restricted'):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        bsz, seq_len, _ = input_ids.size()
        class_mask = torch.ones(bsz, 1, device=input_ids.device)
        img_mask = self.bbox2_mask(bbox, self.image_seq_len).view(bsz, -1)  # [B, sqrt * sqrt]
        if 'restricted' in name:
            returned_attention_mask = torch.cat((class_mask, img_mask, attention_mask), -1).unsqueeze(-1).expand(-1, -1, seq_len,).unsqueeze(1)  # [B, seq, seq]
            # torchvision.utils.save_image(attention_mask[0], "attention_mask_0.jpg", normalize=True)
            # torchvision.utils.save_image(attention_mask[1], "attention_mask_1.jpg", normalize=True)
            maxval = torch.tensor(torch.finfo(input_ids.dtype).max)
            return returned_attention_mask*maxval - maxval + 1.
        elif 'semi' in name:
            semi_attention_mask = torch.ones(bsz, seq_len, seq_len-self.image_seq_len-1, device=input_ids.device)  # [B, seq_len, seq_len - self.image_seq_len]
            returned_attention_mask = torch.cat(
                (torch.cat((class_mask, img_mask, attention_mask), -1).unsqueeze(-1).expand(-1, -1, self.image_seq_len + 1,),
                semi_attention_mask), -1
                ).unsqueeze(1)  # [B, seq_len] -> [B, seq_len, 1] -> [B, seq_len, self.image_seq_len + 1(class)] -> [B, seq_len, self.image_seq_len + 1 + seq_len - self.image_seq_len-1] -> [B, seq_len, seq_len]
            # torchvision.utils.save_image(attention_mask[0], "attention_mask_0.jpg", normalize=True)
            # torchvision.utils.save_image(attention_mask[1], "attention_mask_1.jpg", normalize=True)
            maxval = torch.tensor(torch.finfo(input_ids.dtype).max)
            return returned_attention_mask*maxval - maxval + 1.
        else:
            whole_mask = torch.ones(bsz, self.image_seq_len, self.image_seq_len, device=input_ids.device)
            reversed_mask = 1.-img_mask
            else_img_mask = whole_mask - img_mask.unsqueeze(-1) * reversed_mask.unsqueeze(-2) + img_mask.unsqueeze(-1) * img_mask.unsqueeze(-2)  # [B, sqrt*sqrt, sqrt*sqrt]
            additional_class_mask = class_mask.expand(-1, seq_len-1)
            additional_text_mask = 0.
            returned_attention_mask = torch.cat((
                    torch.cat((
                        class_mask,
                        img_mask,
                        attention_mask
                    ), -1),
                    torch.cat((
                        additional_class_mask[:,:self.image_seq_len*self.image_seq_len-1],
                        else_img_mask,
                        attention_mask.unsqeeze(-1).expand(-1, -1, self.image_seq_len * self.image_seq_len)
                    ), -1),
                    torch.cat((
                        additional_class_mask[:,self.image_seq_len*self.image_seq_len]
                    ), -1)
                )
            )
            return None
    
    def bbox2_mask(self, bbox, sqrt_img_seq_len):  # [B, head, query, key]   --> [B, head, inp_seq_len, 1]
        x1 = torch.floor(bbox[:,0] * sqrt_img_seq_len).unsqueeze(-1)
        y1 = torch.floor(bbox[:,1] * sqrt_img_seq_len).unsqueeze(-1)
        x2 = torch.ceil((bbox[:,0] + bbox[:,2]) * sqrt_img_seq_len).unsqueeze(-1)
        y2 = torch.ceil((bbox[:,1] + bbox[:,3]) * sqrt_img_seq_len).unsqueeze(-1)

        h_linspace = torch.unsqueeze(torch.linspace(0, sqrt_img_seq_len-1, steps=sqrt_img_seq_len), 0).to(device=x1.device) #  [1, H]
        w_linspace = torch.unsqueeze(torch.linspace(0, sqrt_img_seq_len-1, steps=sqrt_img_seq_len), 0).to(device=x1.device)  # [1, W]

        x1_bool = torch.le(torch.tile(x1, (1, sqrt_img_seq_len)), w_linspace)  # [1, W] [B, W]
        x2_bool = torch.le(w_linspace, torch.tile(x2, (1, sqrt_img_seq_len)))  # [1, W] [B, W]
        y1_bool = torch.le(torch.tile(y1, (1, sqrt_img_seq_len)), h_linspace)  # [1, H] [B, H]
        y2_bool = torch.le(h_linspace, torch.tile(y2, (1, sqrt_img_seq_len)))  # [1, H] [B, H]
        
        x_bool = torch.unsqueeze(x1_bool * x2_bool, -2)
        y_bool = torch.unsqueeze(y1_bool * y2_bool, -1)
        mask = (y_bool*x_bool).to(torch.float32)
        
        return mask  # 1 filled, 0 masked


class GuesserModelForSequenceClassification(GuesserModel):
    config_class = GuesserConfig
    base_model_prefix = "guesser"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    
    def __init__(self, 
            config: GuesserConfig,
            *inputs,
            vision_text_pretrained=None,
            vision_model_path=None, 
            language_model_path=None,
            **kwargs):
        super().__init__(config, *inputs, **kwargs)
    
        vision_config = config.vision_text_model_config.vision_config
        image_size = vision_config.image_size
        patch_size = vision_config.patch_size
        self.image_seq_len = (image_size//patch_size)
        self.image_special_tokens_left = 1

        classifier_dropout = 0.
        self.num_labels = num_labels = 81
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.fusion_model_config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        answer = None,
        category = None,
        bbox = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        if 'clip' in self.vision_text_model.base_model_prefix:
                vision_text_output = self.vision_text_model(
                input_ids = input_ids, 
                pixel_values = pixel_values, 
                attention_mask = attention_mask, 
                position_ids = position_ids, 
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states, 
                return_dict = return_dict,
            )
        else: 
            vision_text_output = self.vision_text_model(
                input_ids = input_ids, 
                pixel_values = pixel_values, 
                attention_mask = attention_mask, 
                position_ids = position_ids, 
                token_type_ids = token_type_ids,
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states, 
                return_dict = return_dict,
            )

        vision_seq_embeds = vision_text_output.image_embeds
        text_seq_embeds = vision_text_output.text_embeds
        fusion_model_input_ids = torch.cat((vision_seq_embeds, text_seq_embeds,), 1)
        attention_mask = torch.ones(attention_mask.size(0), 1, device=attention_mask.get_device()) if 'clip' in self.vision_text_model.base_model_prefix else attention_mask
        attention_mask = self._build_our_attention_mask(fusion_model_input_ids, attention_mask, bbox)
        outputs = self.fusion_model(
                input_ids = fusion_model_input_ids, # [B, 1+49+seq, 512]
                attention_mask = attention_mask,
                position_ids = position_ids,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
                return_dict = return_dict,
                answer = answer,
                category = category,
                bbox = bbox
            )

        pooler_output = outputs.pooler_output
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )