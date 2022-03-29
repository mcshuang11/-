# coding=utf8
from transformers import (
    TapasTokenizer,
    TapasForQuestionAnswering
)
from bert4tf.deploy.tapas_question_answering import TapasQuestioningAnswerPipeline
from flask_cors import *
from flask import request, Flask
import logging
import requests
import json
import pandas as pd

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)
TAPAS_URL = "https://pai.tigerobo.com/x-pai-serving/invoke?appId=d536ac901b0444a3be55092915f9a179&apiKey=92556062a0b6578ed1f45105c1919f37&accessToken=2cef83c8fc3aa6891349a2e37117b70c"

logger.info("Loading tapas tokenizer")
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
logger.info("Loading tapas model")
tapas_model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
tapas_pipe = TapasQuestioningAnswerPipeline(tokenizer=tokenizer, model=tapas_model)


@app.route("/infer", methods=["POST", "GET"])
def infer():
    data = request.json
    table = data.get("table")
    queries = data.get("queries")
    if table is None:
        return {
            "status": 1,
            "msg": "没有解析到table, 请检查传入的参数"
        }
    if queries is None:
        return {
            "status": 1,
            "msg": "没有解析到queries, 请检查传入的参数"
        }

    # translate queries and table headers and cells
    queries = text_translation(queries)
    table = {text_translation(key): [text_translation(v) for v in value] for key, value in table.items()}
    table = pd.DataFrame.from_dict(table)

    res = tapas_pipe(table, queries)
    return json.dumps({"status": 0, "result": res})


def text_translation(text):
    text = {"text": text}
    result = requests.post(url=TAPAS_URL, json=text)
    result = result.json()["data"]["result"]
    logger.info("post request to text_translation")
    return result


if __name__ == "__main__":
    app.run('0.0.0.0', port=9510)
