from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

app = Flask(__name__)

# Initialize the OpenAI client with necessary parameters
client = OpenAI(
    api_key="dummy",  # Enter a random string here, as this is a virtual key example
    base_url=PORTKEY_GATEWAY_URL,
    default_headers=createHeaders(
        provider="anthropic",
        api_key="1+YM2sBEgaZWe45bMOq2huic1uyF",  # Your actual Portkey API key
        virtual_key="002-anthropic-89e57f"  # Your actual Anthropic virtual key
    )
)

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    user_message = data.get('message')

    # Create a chat completion request
    chat_complete = client.chat.completions.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=250,
        messages=[{"role": "user", "content": user_message}],
    )

    response_message = chat_complete.choices[0].message.content
    return jsonify({'reply': response_message})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  # Listen on all network interfaces
