# coding=utf8
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bert4tf.deploy.gec import GECPipeline
import json
from flask_cors import *
from flask import request, Flask
import logging
import re
from typing import Dict, Callable


logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, supports_credentials=True)

tokenizer = AutoTokenizer.from_pretrained("/home/shuang.mo/bert4tf/outputs/train_pig_v3")
model = AutoModelForSeq2SeqLM.from_pretrained("/home/shuang.mo/bert4tf/outputs/train_pig_v3")
pipe = GECPipeline(tokenizer=tokenizer, model=model)
zh2en_punc_map = {"，": ",", "：": ":", "！": "!", "？": "?", "；": ";"}
en2zh_punc_map = {v: k for k, v in zh2en_punc_map.items()}


def dict2replace_func(d: Dict[str, str]) -> Callable[[str], str]:
    left = '|'.join(d.keys())
    left = f'({left})'
    regex = re.compile(left)

    def repl(match) -> str:
        return d.get(match.group(0))

    def func(s: str) -> str:
        return regex.sub(repl, s)

    return func

zh2en_punc_func = dict2replace_func(zh2en_punc_map)


@app.route("/infer", methods=["POST", "GET"])
def infer():
    data = request.json
    text_list = data.get("text_list")
    if text_list is None or not text_list:
        result = {
            "status": 1,
            "msg": "没有解析到text_list, 请检查传入的参数"
        }
        return result
    text_list = [zh2en_punc_func(t) for t in text_list]
    gec_res = pipe(text_list)
    return json.dumps({"status": 0, "result": gec_res}, ensure_ascii=False)


if __name__ == "__main__":
    app.run('0.0.0.0', port=9506)
