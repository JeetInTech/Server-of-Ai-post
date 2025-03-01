from flask import Flask, request, redirect, jsonify
from flask_cors import CORS
import os
import base64
import requests
import time
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
# if not os.getenv('TWITTER_CLIENT_ID') or not os.getenv('TWITTER_CLIENT_SECRET') or not os.getenv('TWITTER_REDIRECT_URI'):
#     raise ValueError("Missing Twitter credentials or redirect URI in .env")

app = Flask(__name__)
CORS(app)

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
HF_API_KEY = os.getenv('HF_API_KEY')
HF_TEXT_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'

TWITTER_AUTH_URL = 'https://api.twitter.com/2/oauth2/authorize'
TWITTER_TOKEN_URL = 'https://api.twitter.com/2/oauth2/token'
TWITTER_POST_URL = 'https://api.twitter.com/2/tweets'

TWITTER_CLIENT_ID = os.getenv('TWITTER_CLIENT_ID')
TWITTER_CLIENT_SECRET = os.getenv('TWITTER_CLIENT_SECRET')
TWITTER_REDIRECT_URI = os.getenv('TWITTER_REDIRECT_URI', 'http://localhost:5002/twitter/callback')

# Retry function with exponential backoff
def retry_request(url, data, headers, retries=5, initial_delay=1):
    delay = initial_delay
    for i in range(retries):
        try:
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            time.sleep(delay)
            delay *= 2
    raise Exception("Max retries exceeded")

def generate_text(prompt):
    gemini_prompt = (
        f"Create an engaging Twitter post about '{prompt}' focusing on legal texts, political opinions, global opinions, or climate issues with these requirements:\n"
        "- Length: 280 characters max\n"
        "- Include 1-2 relevant emojis (e.g., ⚖️ for legal, 🌍 for global, 🔥 for climate, 🗳️ for political)\n"
        "- Add 1-2 unique hashtags (e.g., #Law, #Politics, #ClimateChange, #GlobalIssues)\n"
        "- Tone: Informative, concise, and professional\n"
        "- Format: Just the tweet text, no additional explanations\n"
        "Examples:\n"
        "- 'New climate laws in effect! 🌍 Reduce emissions now. #ClimateChange #Sustainability'\n"
        "- 'Global leaders debate trade policies. 🗳️ Stay informed! #Politics #GlobalIssues'\n"
        "- 'Legal ruling on privacy rights today. ⚖️ Check the details. #Law #Privacy'"
    )

    try:
        gemini_response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}',
            json={'contents': [{'parts': [{'text': gemini_prompt}]}]},
            timeout=15
        )
        gemini_response.raise_for_status()
        post_text = gemini_response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        return post_text[:280]  # Ensure max length for Twitter
    except Exception as e:
        print(f"Gemini error: {str(e)}. Falling back to Hugging Face...")
        return generate_with_huggingface(prompt)

def generate_with_huggingface(prompt):
    hf_headers = {'Authorization': f'Bearer {HF_API_KEY}', 'Content-Type': 'application/json'}
    hf_prompt = (
        f"[INST]Generate a Twitter post about '{prompt}' focusing on legal texts, political opinions, global opinions, or climate issues. "
        "Requirements:\n"
        "- Max 280 characters\n"
        "- 1-2 relevant emojis (e.g., ⚖️ for legal, 🌍 for global, 🔥 for climate, 🗳️ for political)\n"
        "- 1-2 unique hashtags (e.g., #Law, #Politics, #ClimateChange, #GlobalIssues)\n"
        "- Informative, concise, and professional tone\n"
        "- No additional explanations\n"
        "Example format:\n"
        "'New climate laws in effect! 🌍 Reduce emissions now. #ClimateChange #Sustainability'\n"
        "[/INST]"
    )

    try:
        response = retry_request(
            f'https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}',
            {'inputs': hf_prompt, 'parameters': {'max_new_tokens': 300, 'temperature': 0.7, 'repetition_penalty': 1.2, 'do_sample': True}},
            hf_headers
        )
        post_text = response[0].get('generated_text', '').strip()
        if '[/INST]' in post_text:
            post_text = post_text.split('[/INST]')[-1].strip()
        return post_text[:280]
    except Exception as e:
        print(f"Hugging Face error: {str(e)}")
        return "New climate action needed! 🌍 #ClimateChange #Sustainability"

# Twitter Authentication Routes
@app.route('/twitter/auth')
def twitter_auth():
    auth_url = (
        f"{TWITTER_AUTH_URL}?response_type=code"
        f"&client_id={TWITTER_CLIENT_ID}"
        f"&redirect_uri={TWITTER_REDIRECT_URI}"
        f"&scope=tweet.write%20users.read%20offline.access"
        f"&state=state"
        f"&code_challenge=challenge"
        f"&code_challenge_method=plain"
    )
    logger.debug(f"Generated auth URL: {auth_url}")
    return redirect(auth_url)
@app.route('/twitter/callback')
def twitter_callback():
    code = request.args.get('code')
    print(f"Received callback with code: {code}")
    client_creds = f"{TWITTER_CLIENT_ID}:{TWITTER_CLIENT_SECRET}"
    encoded_creds = base64.b64encode(client_creds.encode()).decode()
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': f'Basic {encoded_creds}'
    }
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': TWITTER_REDIRECT_URI,
        'code_verifier': 'challenge'  # Must match code_challenge
    }
    try:
        token_response = requests.post(
            TWITTER_TOKEN_URL,
            headers=headers,
            data=data,
            timeout=10
        )
        print(f"Token response status: {token_response.status_code}")
        print(f"Token response text: {token_response.text}")
        token_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_details = {
            "error": "Token exchange failed",
            "status_code": e.response.status_code if e.response else None,
            "response_text": e.response.text if e.response else str(e)
        }
        print(f"Token request failed: {error_details}")
        return jsonify(error_details), 500
    response_data = token_response.json()
    access_token = response_data.get('access_token')
    if not access_token:
        return jsonify({'error': 'No access token received'}), 500
    return redirect(f'http://localhost:3000?token={access_token}&platform=twitter')

# Generate Twitter Post (Text-Only)
@app.route('/twitter/generate-post', methods=['POST'])
def twitter_generate_post():
    prompt = request.json.get('prompt')
    post_text = generate_text(prompt)
    return jsonify({'post': post_text})

# Post to Twitter (Text-Only)
@app.route('/twitter/post', methods=['POST'])
def twitter_post():
    data = request.json
    tweet_text = data.get('tweet_text')
    token = data.get('token')
    headers = {'Authorization': f'Bearer {token}'}
    payload = {'text': tweet_text}
    response = requests.post(TWITTER_POST_URL, headers=headers, json=payload)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(port=5002, debug=True)