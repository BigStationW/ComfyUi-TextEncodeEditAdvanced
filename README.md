# ComfyUi-TextEncodeEditAdvanced

## Intro
Qwen Image Edit uses a Vision Language Model (VLM) to analyze your input images and automatically enhance your prompt with more detailed descriptions.

The default TextEncodeQwenImageEdit node downscales your images to 0.15 megapixels before feeding them to the VLM.

This custom node gives you control over that threshold (vl_megapixels), allowing you to eventually find a better sweet spot for your specific use case.

By adjusting this threshold, you may achieve:

- Better style preservation
- Reduced zoom effect: Mitigate the tendency for Qwen Image Edit to zoom in on images

https://github.com/user-attachments/assets/23dca6a6-6add-44b5-8777-9c206ea66f9f

When going for vl_megapixels = 0, it has the same effect as stacking multiple ReferenceLatent nodes with each other.

## Installation

Navigate to the **ComfyUI/custom_nodes** folder, [open cmd](https://www.youtube.com/watch?v=bgSSJQolR0E&t=47s) and run:

```bash
git clone https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced
```

Restart ComfyUI after installation.


## Usage
Find the **TextEncodeEditAdvanced** node

<img width="500" alt="image" src="https://github.com/user-attachments/assets/e5b6c4f1-f7f8-4679-b62f-d313bb8b537a" />


I also provide [worksflows](https://github.com/BigStationW/ComfyUi-TextEncodeEditAdvanced/tree/main/workflow) for those interested.
