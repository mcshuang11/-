import logging
from typing import List, Dict
from torch import Tensor
import torch
from Levenshtein import opcodes
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

logger = logging.getLogger(__name__)


def get_edit(sources, targets):
    res = []
    for source, target in zip(sources, targets):
        edits = opcodes(source, target)
        new_edits = []
        for edit in edits:
            if edit[0] == "equal":
                continue
            s_start, s_end = int(edit[1]), int(edit[2])
            t_start, t_end = int(edit[3]), int(edit[4])
            if edit[0] == "insert" and target[t_start:t_end] in {"......", "."}:
                continue
            if t_start == len(target)-1 and target[t_start] in {"。", "？", "！", "了", "啦", "吗"}:
                continue
            cur_edit = dict()
            cur_edit["start"] = s_start
            cur_edit["end"] = s_end
            cur_edit["text"] = source[s_start:s_end]
            cur_edit["corrected_text"] = target[t_start:t_end]
            if edit[0] == "insert":
                cur_edit["operation"] = "插入"
                cur_edit["operation_detail"] = f"建议插入 \"{target[t_start:t_end]}\""
            elif edit[0] == "replace":
                cur_edit["operation"] = "替换"
                cur_edit["operation_detail"] = f"建议用 \"{target[t_start:t_end]}\" 替换 \"{source[s_start:s_end]}\""
            elif edit[0] == "delete":
                cur_edit["operation"] = "删除"
                cur_edit["operation_detail"] = f"建议删除 \"{source[s_start:s_end]}\""

            new_edits.append(cur_edit)
        res.append(new_edits)
    return res


chinese_puncs = ['，', '。', '、', '：', '；', '！', '？', '｜', '【', '】', '—', '·', '（', '）', '《', '》', '\xa0']


def filter_chin_puncs(result):
    all_data = []
    for res in result:
        data = []
        for item in res:
            if item['text'] not in chinese_puncs:
                data.append(item)
        all_data.append(data)
    return all_data


class GECPipeline:
    def __init__(self,
                 tokenizer,
                 model,
                 max_source_length: int = 200,
                 max_target_length: int = 200,
                 device=device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        if self.device.type == "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.source_prefix = "gec: "

    def preprocess(self, examples: List[str]) -> Dict[str, Tensor]:
        added_prefix = [self.source_prefix + example for example in examples]
        inputs = self.tokenizer(added_prefix,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_source_length)
        return inputs

    def __call__(self, texts: List[str], *args, **kwargs):
        logger.info("*** Prediction ***")
        inputs = self.preprocess(texts)
        input_ids = inputs["input_ids"]
        input_mask = inputs["attention_mask"]

        if self.device.type == "cuda":
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)

        with torch.no_grad():
            outs = self.model.generate(input_ids=input_ids,
                                       attention_mask=input_mask,
                                       max_length=self.max_target_length,
                                       do_sample=True,
                                       top_k=50,
                                       top_p=0.95,
                                       no_repeat_ngram_size=4,
                                       num_beams=4,
                                       num_return_sequences=1)

        res = self.tokenizer.batch_decode(outs,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        res_edit = get_edit(texts, res)
        return res_edit



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/home/shuang.mo/bert4tf/outputs/train_pig_v2_update1")
    model = AutoModelForSeq2SeqLM.from_pretrained("/home/shuang.mo/bert4tf/outputs/train_pig_v2_update1")
    pipe = GECPipeline(tokenizer=tokenizer, model=model)
    # 如果美不俄插手，以色列和伊朗谁的胜算高？
    texts = [
        '温州三家人：老太太欲要陪女儿一段时间，老头心疼她勉强同意', '华胥引：都督连夜赶回，竟是为质问夫人，夫人心痛不已',
        '女孩读过爸爸写的小说，对自己的梦想产生质疑','班长也有不为人知的一面，表面光鲜亮丽，实际家庭困难',
        '班长给同学信纸，让他和自己交换，这会不会不太好','我以为我的温柔能治愈你，却没想到你竟然被伤得那么深'

    ]
    res = filter_chin_puncs(pipe(texts))
    print(res)
    # print(get_edit(texts, res))
    # import ipdb;ipdb.set_trace()
