from flask import Flask, jsonify, request
from flask_cors import CORS
import ldclient
from ldclient import Context
from ldclient.config import Config
from ldai.client import LDAIClient, AIConfig, ModelConfig, LDMessage, ProviderConfig
import pprint
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

ldclient.set_config(Config(os.environ["LD_SDK_KEY"]))
aiclient = LDAIClient(ldclient.get())
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

app = Flask(__name__)
CORS(app, origins=["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:5173", "http://127.0.0.1:5173"])

@app.route('/context', methods=['POST'])
def create_context():
    data = request.get_json()
    multi_context = Context.from_dict(data)
    ai_config_key = "the-voice"

    default_value = AIConfig(
        enabled=True,
        model=ModelConfig(name='my-default-model'),
        messages=[],
    )

    config, tracker = aiclient.config(
        ai_config_key,
        multi_context,
        default_value
    )

    app.logger.info(f"Config value: {config}")
    app.logger.info(f"Model config: {pprint.pformat(getattr(config, 'model', None).__dict__ if getattr(config, 'model', None) else None)}")
    app.logger.info(f"Provider: {pprint.pformat(getattr(config, 'provider', None).__dict__ if getattr(config, 'provider', None) else None)}")
    app.logger.info(f"Messages: {getattr(config, 'messages', None)}")

    messages = [] if config.messages is None else config.messages
    completion = tracker.track_openai_metrics(
        lambda:
            openai_client.chat.completions.create(
                model=config.model.name,
                messages=[message.to_dict() for message in messages],
        )
    )
    return completion.choices[0].message.content, 200

if __name__ == '__main__':
    app.run(debug=True)
