import math
import torch
import comfy.utils
from comfy_extras import nodes_custom_sampler
from nodes import node_helpers

class TextEncodeQwenImageEditAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "vl_megapixels": ("FLOAT", {
                    "default": 0.50, 
                    "min": 0.01, 
                    "max": 4.0, 
                    "step": 0.01,
                    "display": "number",
                    "tooltip": "Target megapixels for Vision-Language model. Recommended: 0.2-1.0 MP. Qwen2.5-VL trained range: 0.2-1.0 MP"
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

    def encode(self, clip, prompt, vl_megapixels=0.50, vae=None, image1=None, image2=None, image3=None):
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""
        
        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                print(f"\n=== Image {i+1} Processing (Advanced) ===")
                print(f"Original dimensions: {samples.shape[3]}×{samples.shape[2]}")
                
                # VL (Vision-Language) resize - configurable megapixels
                total = int(vl_megapixels * 1_000_000)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                print(f"VL resize - Target: {vl_megapixels} MP ({total} pixels)")
                print(f"VL resize - Scale factor: {scale_by:.4f}")
                print(f"VL resize - Result: {width}×{height}")
                
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                
                if vae is not None:
                    # NO VAE resize - encode at original resolution
                    ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))
                    
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
        
        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        return (conditioning,)

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "TextEncodeQwenImageEditAdvanced": TextEncodeQwenImageEditAdvanced,
}

# Node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeQwenImageEditAdvanced": "TextEncodeQwenImageEditAdvanced",
}
