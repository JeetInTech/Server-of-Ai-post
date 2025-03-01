from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
import base64
import time
from PIL import Image
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HF_API_KEY = os.getenv('HF_API_KEY')
HF_TEXT_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'
HF_IMAGE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'
FALLBACK_IMAGE_MODEL = 'stabilityai/sdxl-turbo'

# Directory to save files (relative to project root)
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'instagram_posts')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# Retry function with exponential backoff
def retry_request(url, data, headers, retries=5, initial_delay=1, response_type='json'):
    delay = initial_delay
    for i in range(retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            return response.content if response_type == 'arraybuffer' else response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}, retry {i+1}/{retries}")
            time.sleep(delay)
            delay *= 2
    raise Exception("Max retries exceeded")

def generate_text(prompt):
    gemini_prompt = (
        f"Create an engaging Instagram Story text post about '{prompt}' with these requirements:\n"
        "- Length: Up to 2,200 characters (Instagram Story text limit)\n"
        "- Include 1-3 relevant emojis (e.g., 🌟, 📸, ✨, 🎉)\n"
        "- Add 2-3 unique hashtags (e.g., #InstaGood, #StoryTime, #Explore)\n"
        "- Tone: Casual, engaging, and trendy\n"
        "- Structure: Catchy opening, key details, call-to-action (e.g., 'Swipe up!')\n"
        "- Format: Plain text, no markdown\n"
        "Examples:\n"
        "- '🌟 Capturing the vibe today! 📸 Check out this amazing moment—pure inspiration! ✨ #InstaGood #StoryTime #Explore'\n"
    )

    try:
        gemini_response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}',
            json={'contents': [{'parts': [{'text': gemini_prompt}]}]},
            timeout=15
        )
        gemini_response.raise_for_status()
        post_text = gemini_response.json()['candidates'][0]['content']['parts'][0]['text'].strip()[:2200]
        return post_text
    except Exception as e:
        print(f"Gemini error: {e}. Falling back to Hugging Face...")
        return generate_with_huggingface(prompt)

def generate_with_huggingface(prompt):
    hf_headers = {'Authorization': f'Bearer {HF_API_KEY}', 'Content-Type': 'application/json'}
    hf_prompt = (
        f"[INST]Generate an Instagram Story text post about '{prompt}'. "
        "Requirements:\n"
        "- Up to 2,200 characters\n"
        "- 1-3 relevant emojis (e.g., 🌟, 📸, ✨, 🎉)\n"
        "- 2-3 unique hashtags (e.g., #InstaGood, #StoryTime, #Explore)\n"
        "- Casual, engaging, trendy tone\n"
        "- Catchy opening, key details, call-to-action (e.g., 'Swipe up!')\n"
        "- No markdown\n"
        "Example: '🌟 Capturing the vibe today! 📸 Check out this moment! ✨ #InstaGood #StoryTime #Explore' [/INST]"
    )

    try:
        response = retry_request(
            f'https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}',
            {'inputs': hf_prompt, 'parameters': {'max_new_tokens': 2500, 'temperature': 0.7, 'repetition_penalty': 1.2, 'do_sample': True}},
            hf_headers
        )
        post_text = response[0].get('generated_text', '').strip()
        if '[/INST]' in post_text:
            post_text = post_text.split('[/INST]')[-1].strip()[:2200]
        return post_text
    except Exception as e:
        print(f"Hugging Face error: {e}")
        return "🌟 New vibes incoming! 📸 Check this out! ✨ #InstaGood #StoryTime #Explore"

def generate_image(prompt):
    try:
        image_prompt = (
            f"Create a vibrant Instagram Story image for '{prompt}'. "
            f"Style: Modern, colorful, trendy. Dimensions: 1080x1920 pixels. Avoid text or logos. Focus on: Lifestyle, events, or abstract visuals."
        )
        gemini_response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}',
            json={'contents': [{'parts': [{'text': image_prompt}]}]},
            timeout=15
        )
        gemini_response.raise_for_status()
        description = gemini_response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"Gemini error: {e}. Falling back to original prompt.")
        description = prompt

    hf_headers = {'Authorization': f'Bearer {HF_API_KEY}', 'Content-Type': 'application/json'}
    try:
        image_response = retry_request(
            f'https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}',
            {'inputs': description, 'parameters': {'width': 1080, 'height': 1920, 'num_inference_steps': 30, 'guidance_scale': 7.5}},
            hf_headers,
            response_type='arraybuffer'
        )
        img = Image.open(io.BytesIO(image_response))
        img.save(os.path.join(SAVE_DIR, f"{int(time.time())}_image.png"))
        return os.path.join(SAVE_DIR, f"{int(time.time())}_image.png")
    except Exception as e:
        print(f"Primary image model failed: {e}. Switching to fallback...")
        image_response = requests.post(
            f'https://api-inference.huggingface.co/models/{FALLBACK_IMAGE_MODEL}',
            json={'inputs': description, 'parameters': {'width': 1080, 'height': 1920, 'num_inference_steps': 10, 'guidance_scale': 5.0}},
            headers=hf_headers,
            timeout=60
        ).content
        img = Image.open(io.BytesIO(image_response))
        img.save(os.path.join(SAVE_DIR, f"{int(time.time())}_image_fallback.png"))
        return os.path.join(SAVE_DIR, f"{int(time.time())}_image_fallback.png")

@app.route('/instagram/generate-story', methods=['POST'])
def generate_instagram_story():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Generate text and image
    post_text = generate_text(prompt)
    image_path = generate_image(prompt)

    # Save text to a file
    text_filename = os.path.join(SAVE_DIR, f"{int(time.time())}_text.txt")
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(post_text)

    return jsonify({
        'text_file': text_filename,
        'image_file': image_path,
        'message': 'Files generated and saved. Upload to Instagram Story manually.'
    })

if __name__ == '__main__':
    app.run(port=5003, debug=True)