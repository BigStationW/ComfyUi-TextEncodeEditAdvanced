import math
import torch
import comfy.utils
from nodes import node_helpers

class ConditioningAddImageReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "max_images_allowed": (["0", "1", "2", "3"], {
                    "default": "3",
                    "tooltip": "Maximum number of images to process. Images are processed in order: image1, image2, image3"
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "add_references"
    CATEGORY = "conditioning/qwen_image_edit"
    
    def add_references(self, conditioning, max_images_allowed="3", vae=None, image1=None, image2=None, image3=None):
        # Convert string to int
        max_images_allowed = int(max_images_allowed)
        
        ref_latents = []
        images = [image1, image2, image3]
        
        # Only process up to max_images_allowed
        for i, image in enumerate(images[:max_images_allowed]):
            if image is not None and vae is not None:
                samples = image.movedim(-1, 1)
                ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))
        
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        return (conditioning,)


class TextEncodeEditAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "vl_megapixels": ("FLOAT", {
                    "default": 0.50, 
                    "min": 0.0,
                    "max": 4.0, 
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Target megapixels for Vision-Language model. Set to 0 to disable VL image feeding. Recommended: 0.2-1.0 MP. Qwen2.5-VL trained range: 0.2-1.0 MP"
                }),
                "max_images_allowed": (["0", "1", "2", "3"], {
                    "default": "3",
                    "tooltip": "Maximum number of images to process. Images are processed in order: image1, image2, image3"
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/qwen_image_edit"
    
    def encode(self, clip, prompt, vl_megapixels=0.50, max_images_allowed="3", vae=None, image1=None, image2=None, image3=None):
        # Convert string to int
        max_images_allowed = int(max_images_allowed)
        
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""
        
        vl_disabled = vl_megapixels <= 0
        
        # Only process up to max_images_allowed
        for i, image in enumerate(images[:max_images_allowed]):
            if image is not None:
                samples = image.movedim(-1, 1)
                
                if not vl_disabled:
                    total = int(vl_megapixels * 1_000_000)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    
                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    images_vl.append(s.movedim(1, -1))
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                
                if vae is not None:
                    ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))
        
        tokens = clip.tokenize(
            image_prompt + prompt,
            images=images_vl if not vl_disabled and len(images_vl) > 0 else None,
            llama_template=llama_template if not vl_disabled else None
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        return (conditioning,)


class ConditioningAddImageReferenceDual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "max_images_allowed": (["0", "1", "2", "3"], {
                    "default": "3",
                    "tooltip": "Maximum number of images to process. Images are processed in order: image1, image2, image3"
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "add_references"
    CATEGORY = "conditioning/qwen_image_edit"
    
    def add_references(self, positive, negative, max_images_allowed="3", vae=None, image1=None, image2=None, image3=None):
        # Convert string to int
        max_images_allowed = int(max_images_allowed)
        
        ref_latents = []
        images = [image1, image2, image3]
        
        # Only process up to max_images_allowed
        for i, image in enumerate(images[:max_images_allowed]):
            if image is not None and vae is not None:
                samples = image.movedim(-1, 1)
                ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))
        
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        
        return (positive, negative)


class TextEncodeEditAdvancedDual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "vl_megapixels": ("FLOAT", {
                    "default": 0.50, 
                    "min": 0.0,
                    "max": 4.0, 
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Target megapixels for Vision-Language model. Set to 0 to disable VL image feeding. Recommended: 0.2-1.0 MP. Qwen2.5-VL trained range: 0.2-1.0 MP"
                }),
                "max_images_allowed": (["0", "1", "2", "3"], {
                    "default": "3",
                    "tooltip": "Maximum number of images to process. Images are processed in order: image1, image2, image3"
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "conditioning/qwen_image_edit"
    
    def encode(self, clip, positive, negative, vl_megapixels=0.50, max_images_allowed="3", vae=None, image1=None, image2=None, image3=None):
        # Convert string to int
        max_images_allowed = int(max_images_allowed)
        
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""
        
        vl_disabled = vl_megapixels <= 0
        
        # Only process up to max_images_allowed
        for i, image in enumerate(images[:max_images_allowed]):
            if image is not None:
                samples = image.movedim(-1, 1)
                
                if not vl_disabled:
                    total = int(vl_megapixels * 1_000_000)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    
                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    images_vl.append(s.movedim(1, -1))
                    image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                
                if vae is not None:
                    ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))
        
        # Encode positive
        tokens_positive = clip.tokenize(
            image_prompt + positive,
            images=images_vl if not vl_disabled and len(images_vl) > 0 else None,
            llama_template=llama_template if not vl_disabled else None
        )
        conditioning_positive = clip.encode_from_tokens_scheduled(tokens_positive)
        
        # Encode negative
        tokens_negative = clip.tokenize(
            negative,
            images=None,
            llama_template=None
        )
        conditioning_negative = clip.encode_from_tokens_scheduled(tokens_negative)
        
        if len(ref_latents) > 0:
            conditioning_positive = node_helpers.conditioning_set_values(conditioning_positive, {"reference_latents": ref_latents}, append=True)
            conditioning_negative = node_helpers.conditioning_set_values(conditioning_negative, {"reference_latents": ref_latents}, append=True)
        
        return (conditioning_positive, conditioning_negative)


NODE_CLASS_MAPPINGS = {
    "TextEncodeEditAdvanced": TextEncodeEditAdvanced,
    "ConditioningAddImageReference": ConditioningAddImageReference,
    "ConditioningAddImageReferenceDual": ConditioningAddImageReferenceDual,
    "TextEncodeEditAdvancedDual": TextEncodeEditAdvancedDual,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeEditAdvanced": "TextEncodeEditAdvanced",
    "ConditioningAddImageReference": "Conditioning Add Image Reference",
    "ConditioningAddImageReferenceDual": "Conditioning Add Image Reference (Dual)",
    "TextEncodeEditAdvancedDual": "TextEncodeEditAdvanced (Dual)",
}
