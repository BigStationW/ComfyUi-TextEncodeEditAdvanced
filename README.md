# ComfyUi-TextEncodeQwenImageEditAdvanced

## Intro
Qwen Image Edit uses a Vision Language Model (VLM) to analyze your input images and automatically enhance your prompt with more detailed descriptions.

The default TextEncodeQwenImageEdit node downscales your images to 0.15 megapixels before feeding them to the VLM.

This custom node gives you control over that threshold (vl_megapixels), allowing you to eventually find a better sweet spot for your specific use case.

By adjusting this threshold, you may achieve:

- Better style preservation
- Reduced zoom effect: Mitigate the tendency for Qwen Image Edit to zoom in on images

https://github.com/user-attachments/assets/23dca6a6-6add-44b5-8777-9c206ea66f9f

## Installation

Navigate to the **ComfyUI/custom_nodes** folder, [open cmd](https://www.youtube.com/watch?v=bgSSJQolR0E&t=47s) and run:

```bash
git clone https://github.com/BigStationW/ComfyUi-TextEncodeQwenImageEditAdvanced
```

Restart ComfyUI after installation.


## Usage
Find the **TextEncodeQwenImageEditAdvanced** node

<img width="500" alt="image" src="https://github.com/user-attachments/assets/d8cae86d-bc5e-42ee-a04f-abbe5380effd" />

I also provide [a workflow](https://github.com/BigStationW/ComfyUi-TextEncodeQwenImageEditAdvanced/blob/main/workflow/workflow_Qwen_Image_Edit.json) for those interested.
