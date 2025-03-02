from flask import Flask, request, redirect, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests
import base64
import time
import io
import re
from PIL import Image

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Your Gemini API key
HF_API_KEY = os.getenv('HF_API_KEY')  # Hugging Face API key
HF_TEXT_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'  # More powerful LLM for text
HF_IMAGE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'  # High-quality SDXL for images
FALLBACK_IMAGE_MODEL = 'stabilityai/sdxl-turbo'  # Fast fallback with decent quality

LINKEDIN_AUTH_URL = 'https://www.linkedin.com/oauth/v2/authorization'
LINKEDIN_TOKEN_URL = 'https://www.linkedin.com/oauth/v2/accessToken'
CLIENT_ID = os.getenv('LINKEDIN_CLIENT_ID')
CLIENT_SECRET = os.getenv('LINKEDIN_CLIENT_SECRET')
REDIRECT_URI = os.getenv('LINKEDIN_REDIRECT_URI', 'http://localhost:5001/auth/linkedin/callback')

# Retry function with exponential backoff
def retry_request(url, data, headers, retries=5, initial_delay=1, response_type='json'):
    delay = initial_delay
    for i in range(retries):
        try:
            if response_type == 'arraybuffer':
                response = requests.post(url, json=data, headers=headers, timeout=60)
                response.raise_for_status()
                return response.content
            else:
                response = requests.post(url, json=data, headers=headers, timeout=60)
                response.raise_for_status()
                return response.json()
        except requests.RequestException as e:
            if hasattr(e.response, 'status_code') and e.response.status_code == 503 and i < retries - 1:
                print(f"Retrying ({i + 1}/{retries}) after 503 error, waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
                continue
            raise e
@app.route('/')
def home():
    return "LinkedIn Post Generator API is running. Use /generate-post or /generate-image."
# LinkedIn Authentication Routes
@app.route('/auth/linkedin')
def auth_linkedin():
    # Updated scope without w_organization_social
    auth_url = f"{LINKEDIN_AUTH_URL}?response_type=code&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}&scope=openid%20profile%20email%20w_member_social"
    return redirect(auth_url)

@app.route('/auth/linkedin/callback')
def auth_linkedin_callback():
    error = request.args.get('error')
    if error:
        if error == 'unauthorized_scope_error':
            return jsonify({
                'error': 'unauthorized_scope_error',
                'message': 'A requested scope is not authorized. Please check your app settings in the LinkedIn Developer Portal.'
            }), 400
        return jsonify({'error': error, 'message': 'Authentication failed'}), 400

    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    
    try:
        response = requests.post(LINKEDIN_TOKEN_URL, data={
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET,
        })
        response.raise_for_status()
        access_token = response.json()['access_token']
        return redirect(f'http://localhost:3000?token={access_token}')
    except Exception as e:
        print(f"Auth error: {str(e)}")
        return jsonify({'error': 'Authentication failed', 'details': str(e)}), 500

# Generate LinkedIn Post
@app.route('/generate-post', methods=['POST'])
def generate_post():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Enhanced Gemini prompt with stricter requirements
    gemini_prompt = (
        f"Create a professional LinkedIn post about '{prompt}' with these requirements:\n"
        "- Length: 200-280 characters (MUST meet this)\n"
        "- Include 1-3 relevant emojis (e.g., 🤝, 😊, 🚀, ❤️, MUST include at least one)\n"
        "- Add 2-3 unique, industry-specific hashtags (no repetition, e.g., #Sales #LinkedIn #Growth, MUST include at least two)\n"
        "- Structure: Engaging opening, valuable insight, clear call-to-action\n"
        "- Tone: Professional but approachable\n"
        "- Format: Just the post text, no additional explanations or markdown\n"
        "- Ensure strict adherence to all requirements"
    )

    try:
        gemini_response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}',
            json={
                'contents': [{
                    'parts': [{'text': gemini_prompt}]
                }]
            },
            timeout=15
        )
        gemini_response.raise_for_status()
        post_text = gemini_response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        print(f"Gemini generated post: '{post_text}'")
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]'  # Combined valid emoji ranges
        print(f"Length: {len(post_text)}, Emojis: {re.search(emoji_pattern, post_text) is not None}, Hashtags: {post_text.count('#')}")
        return handle_post_validation(post_text)

    except Exception as e:
        print(f"Gemini error: {str(e)}. Falling back to Hugging Face...")
        return generate_with_huggingface(prompt)

def handle_post_validation(post_text):
    """Clean and enhance posts that fail validation"""
    # Clean up text
    clean_text = re.sub(r'\*\*|__|\[.*?\]', '', post_text)  # Remove markdown
    clean_text = clean_text.strip()
    
    # Add missing emojis if needed
    emoji_candidates = ['🚀', '💡', '👔', '🌍', '📈', '🤝', '🎯', '💼']
    if not re.search(r'[\U0001F600-\U0001F6FF]', clean_text):
        clean_text = f"{emoji_candidates[0]} {clean_text}"
    
    # Add hashtags if missing
    if clean_text.count('#') < 2:
        hashtags = ['ProfessionalGrowth', 'CareerDevelopment', 'IndustryInsights']
        clean_text += f" #{hashtags[0]} #{hashtags[1]}"  # Add two hashtags if less than 2
    
    # Ensure length
    if len(clean_text) > 280:
        clean_text = clean_text[:275] + '...'
    
    # Final validation check
    if not is_valid_post(clean_text):
        print(f"Validation failed after enhancement: '{clean_text}'")
        return jsonify({
            'error': 'Post still doesn\'t meet requirements after enhancement',
            'post': clean_text
        }), 400
    
    return jsonify({'post': clean_text})

def generate_with_huggingface(prompt):
    # Enhanced prompt for Hugging Face with stricter requirements
    hf_prompt = (
        f"[INST]Generate a professional LinkedIn post about '{prompt}'. "
        "Requirements:\n"
        "- 200-280 characters (MUST meet this)\n"
        "- 1-3 relevant emojis (use 🤝, 😊, 🚀, ❤️, MUST include at least one)\n"
        "- 2-3 unique hashtags (no repetition, e.g., #Sales #LinkedIn #Growth, MUST include at least two)\n"
        "- Professional, approachable tone\n"
        "- Engaging opening, valuable insight, clear call-to-action\n"
        "- No additional explanations or markdown\n"
        "Example format:\n"
        "Excited about [topic]! 🚀 Gain insights & connect. #Networking #Growth #Professional [/INST]"
    )
    
    hf_headers = {
        'Authorization': f'Bearer {HF_API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        response = retry_request(
            f'https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}',
            {
                'inputs': hf_prompt,
                'parameters': {
                    'max_new_tokens': 300,
                    'temperature': 0.7,
                    'repetition_penalty': 1.5,
                    'do_sample': True,
                    'return_full_text': False
                }
            },
            hf_headers
        )

        post_text = response[0].get('generated_text', '').strip()
        if '[/INST]' in post_text:
            post_text = post_text.split('[/INST]')[-1].strip()
        print(f"Hugging Face generated post: '{post_text}'")
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF]'  # Same fix here
        print(f"Length: {len(post_text)}, Emojis: {re.search(emoji_pattern, post_text) is not None}, Hashtags: {post_text.count('#')}")
        return handle_post_validation(post_text)

    except Exception as e:
        print(f"Hugging Face error: {str(e)}")
        return jsonify({
            'error': 'Failed to generate post',
            'details': str(e),
            'post': "Excited about new opportunities! 🚀 Connect & grow together. #Networking #Growth #Professional"
        }), 503
def is_valid_post(text):
    """More flexible validation for generated posts"""
    if not text:
        return False
    # Adjust length to match prompt (200-280)
    if len(text) < 200 or len(text) > 300:  # Allow slight overflow
        return False
    # Check for at least one emoji using Unicode emoji detection
    emoji_pattern = re.compile(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+', 
        flags=re.UNICODE
        )
    if not emoji_pattern.search(text):
        return False
    # # Check for at least two hashtags to match the prompt's requirement
    # if text.count('#') < 2:
    #     return False
    return True

# Generate Image (Enhanced with SDXL and fallback)
@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.get_json()
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        # Use Gemini to generate a detailed, high-quality image prompt
        image_prompt = (f"Create a detailed prompt for a LinkedIn post image about '{prompt}'. "
                       f"Visual style: Corporate, modern, vibrant colors. Avoid text or logos. "
                       f"Focus on: Abstract concepts, professional growth, or industry-specific visuals.")
        
        gemini_response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}',
            json={
                'contents': [{
                    'parts': [{'text': image_prompt}]
                }]
            },
            timeout=15
        )
        gemini_response.raise_for_status()
        description = gemini_response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"Gemini error: {str(e)}. Falling back to original prompt.")
        description = prompt

    # Fallback to Hugging Face image generation with optimized SDXL parameters
    hf_headers = {
        'Authorization': f'Bearer {HF_API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        # Use SDXL base 1.0 for high quality
        image_response = retry_request(
            f'https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}',
            {
                'inputs': description,
                'parameters': {
                    'negative_prompt': 'blurry, lowres, text, watermark, disfigured, amateurish',
                    'num_inference_steps': 30,
                    'guidance_scale': 7.5,
                    'width': 1024,
                    'height': 1024,
                    'enhance_prompt': True
                }
            },
            hf_headers,
            response_type='arraybuffer'
        )
    except Exception as e:
        print(f"Primary image model failed: {str(e)}. Switching to fallback...")
        # Use SDXL-Turbo as fallback with optimized parameters
        image_response = requests.post(
            f'https://api-inference.huggingface.co/models/{FALLBACK_IMAGE_MODEL}',
            json={
                'inputs': description,
                'parameters': {
                    'num_inference_steps': 10,
                    'guidance_scale': 5.0,
                    'width': 1024,
                    'height': 1024
                }
            },
            headers=hf_headers,
            timeout=60
        ).content

    # Validate image
    try:
        Image.open(io.BytesIO(image_response)).verify()
        base64_image = base64.b64encode(image_response).decode('utf-8')
        image_url = f'data:image/png;base64,{base64_image}'
        return jsonify({'imageUrl': image_url})
    except Exception as e:
        print(f"Invalid image: {str(e)}")
        return jsonify({
            'imageUrl': 'https://via.placeholder.com/1024x1024.png?text=Image+Generation+Failed'
        }), 503

# Post to LinkedIn
@app.route('/post-to-linkedin', methods=['POST'])
def post_to_linkedin():
    data = request.get_json()
    token = data.get('token')
    post = data.get('post')
    image_url = data.get('imageUrl')

    if not token or not post:
        return jsonify({'error': 'Token and post content are required'}), 400

    try:
        # Add LinkedIn API version header
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
            'X-Restli-Protocol-Version': '2.0.0'
        }

        # Get user info
        user_info_response = requests.get('https://api.linkedin.com/v2/userinfo', headers=headers)
        user_info_response.raise_for_status()
        person_urn = f"urn:li:person:{user_info_response.json()['sub']}"

        # Prepare post data
        post_data = {
            'author': person_urn,
            'lifecycleState': 'PUBLISHED',
            'specificContent': {
                'com.linkedin.ugc.ShareContent': {
                    'shareCommentary': {'text': post},
                    'shareMediaCategory': 'IMAGE' if image_url and 'placeholder' not in image_url else 'NONE',
                },
            },
            'visibility': {'com.linkedin.ugc.MemberNetworkVisibility': 'PUBLIC'}
        }

        # Handle image upload if present
        if image_url and 'placeholder' not in image_url:
            try:
                # Register upload
                register_response = requests.post(
                    'https://api.linkedin.com/v2/assets?action=registerUpload',
                    json={
                        'registerUploadRequest': {
                            'recipes': ['urn:li:digitalmediaRecipe:feedshare-image'],
                            'owner': person_urn,
                            'serviceRelationships': [{
                                'relationshipType': 'OWNER', 
                                'identifier': 'urn:li:userGeneratedContent'
                            }]
                        }
                    },
                    headers=headers
                )
                register_response.raise_for_status()
                upload_url = register_response.json()['value']['uploadMechanism']['com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest']['uploadUrl']
                asset_urn = register_response.json()['value']['asset']

                # Upload image
                image_data = base64.b64decode(image_url.split(',')[1])
                requests.put(upload_url, data=image_data, headers={'Authorization': f'Bearer {token}', 'Content-Type': 'image/png'}).raise_for_status()

                # Add media to post
                post_data['specificContent']['com.linkedin.ugc.ShareContent']['media'] = [
                    {
                        'status': 'READY',
                        'description': {'text': 'Generated Image'},
                        'media': asset_urn,
                        'title': {'text': 'Image Title'}
                    }
                ]
            except Exception as e:
                print(f"Image upload failed: {str(e)}")
                post_data['specificContent']['com.linkedin.ugc.ShareContent']['shareMediaCategory'] = 'NONE'

        # Post to LinkedIn
        response = requests.post('https://api.linkedin.com/v2/ugcPosts', json=post_data, headers=headers)
        response.raise_for_status()

        return jsonify({'success': True, 'postId': response.headers.get('x-restli-id')})
    except Exception as e:
        print(f"LinkedIn posting error: {str(e)}")
        return jsonify({'error': 'Failed to post to LinkedIn', 'details': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))  # Use Render's PORT variable
    app.run(host='0.0.0.0', port=port, debug=False)  # Turn off debug mode in production
