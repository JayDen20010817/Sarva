import os
import json
import openai
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch

# 设置API KEY
openai.api_key = "YOUR_OPENAI_API_KEY"


TEXT_FOLDER = "path/to/text_folder"  
IMAGE_FOLDER = "path/to/image_folder" 
OUTPUT_FOLDER = "path/to/output_folder"  


device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)


sd_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
sd_pipeline = sd_pipeline.to(device)


def polish_text_with_gpt3(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Please polish the given text for clarity and fluency."},
                {"role": "user", "content": text}
            ]
        )
        polished_text = response['choices'][0]['message']['content']
        return polished_text.strip()
    except Exception as e:
        print(f"GPT-3.5 API Error: {e}")
        return text  


def generate_image_caption(image_path):
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = blip_processor(raw_image, return_tensors="pt").to(device)
        caption = blip_model.generate(**inputs)
        caption_text = blip_processor.decode(caption[0], skip_special_tokens=True)
        return caption_text
    except Exception as e:
        print(f"BLIP Error: {e}")
        return ""


def generate_image_with_caption(caption, output_path):
    try:
        image = sd_pipeline(caption).images[0]
        image.save(output_path)
        print(f"Generated image saved to {output_path}")
    except Exception as e:
        print(f"Diffusion Model Error: {e}")


def process_dataset():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for json_file in os.listdir(TEXT_FOLDER):
        if json_file.endswith(".json"):
            with open(os.path.join(TEXT_FOLDER, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            
            text = data.get("text", "")
            image_id = str(data.get("image_id", ""))
            image_path = os.path.join(IMAGE_FOLDER, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                print(f"Image {image_path} not found, skipping...")
                continue

           
            polished_text = polish_text_with_gpt3(text)

            
            caption = generate_image_caption(image_path)
            if not caption:
                print(f"Failed to generate caption for {image_id}, skipping...")
                continue
            
            
            output_image_path = os.path.join(OUTPUT_FOLDER, f"generated_{image_id}.png")
            generate_image_with_caption(caption, output_image_path)

            
            output_data = {
                "image_id": image_id,
                "original_text": text,
                "polished_text": polished_text,
                "generated_caption": caption,
                "generated_image_path": output_image_path
            }
            output_json_path = os.path.join(OUTPUT_FOLDER, f"enhanced_{image_id}.json")
            with open(output_json_path, 'w', encoding='utf-8') as out_f:
                json.dump(output_data, out_f, indent=4, ensure_ascii=False)

            print(f"Processed {json_file} successfully.")