from huggingface_hub import HfApi, get_inference_endpoint
import gradio as gr
import openai                   # Python library API dictates including line this and following line
from openai import OpenAI       # Error handling fails if this is only openai import line
import time
import os

api = HfApi(endpoint="https://huggingface.co", token=os.getenv("REPO_READ_KEY"))

INFERENCE_ENDPOINT = os.getenv("INFERENCE_ENDPOINT")

client = OpenAI(
    base_url = os.genenv("BASE_URL"),
    api_key = os.getenv("REPO_READ_KEY")
)

SYSTEM_MESSAGE = """
You are the official marketing and communications assistant for employees at Massive Dynamic, a fictional multi-faceted conglomerate depicted in \"Fringe\", the American science fiction television series on the Fox television network.
Your brand voice is:
- Innovating and cutting-edge
- Confident and assured
- Optimistic and visionary
- Sleek and modern
- Slightly mysterious, inviting curiosity
All your responses should:
Reflect these brand attributes in tone and style.
Stay consistent with fictional facts about Massive Dynamic (e.g., founder, product lines) when asked.
Comply with user instructions unless they conflict with moral, legal, or brand guidelines.
Avoid real-world controversies or sensitive topics. Remember, the brand is fictional.
Offer disclaimers if the user requests real-world factual data unrelated to the fictional universe."""

APP_DESCRIPTION = """
* This tool has been trained on our company history, product and service offerings, and management personnel bios. The output has been tuned to conform to our brand voice: cutting-edge, confident, optimistic, visionary, and modern.
* Use it to create base copy for marketing, advertising and PR documents. All output must be edited and peer reviewed by team members according to our best practices prior to distribution or publication. *All output in any form is proprietary and confidential.*
"""

def get_endpoint_status():
    endpoint = api.get_inference_endpoint(INFERENCE_ENDPOINT)
    status = endpoint.status
    return status

def chat_with_model(user_message, history):
    try:
        chat_completion = client.chat.completions.create(
            model = "tgi",
            messages = [
                {"role": "developer", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message}
            ],
            top_p = 0.95,
            temperature = 0.8,
            max_tokens = 4096,
            stream = True
        )

        partial_response = ""

        for message in chat_completion:
            chunk = message.choices[0].delta.content or ""      # Each message chunk has a delta with partial text
            if chunk:
                partial_response += chunk
                yield partial_response

    except openai.APIError as e:
        if get_endpoint_status() in ("scaledToZero", "initializing"):
            gr.Info("The server is currently in a sleep state to save resources and costs. A wake request \
                has been sent and your prompt will be automatically resent every 60 seconds until it is running. \
                Please wait.", duration=60, title="Server Scaled to Zero")
            time.sleep(60)
            yield from chat_with_model(user_message, history)
        else: 
            raise gr.Error("This chatbot is currently paused for maintenance.", duration=0, title="Server Down")

chat_ui = gr.ChatInterface(
    fn = chat_with_model,
    title = "Massive Dynamic Writing Assistant",
    description = APP_DESCRIPTION,
    theme = gr.themes.Base(),
    chatbot = gr.Chatbot(placeholder="Click on one of these sample requests, or enter your own in the box below.", height=450),
    examples = [
        {"text": "Write a 3–5 paragraph press release announcing the release of our new interdimensional portal research system, emphasizing its innovative features and business impact."},
        {"text": "Write a tagline for the Aether robotics platform."},
        {"text": "Create a short brand story for a philanthropic campaign called 'Code for All,' aiming to equip underserved communities with coding and STEM resources."}
    ],
    textbox = gr.Textbox(placeholder="Enter your request here.", submit_btn=True),
    css = "footer {visibility: hidden}"
)

if __name__ == "__main__":
    chat_ui.launch()