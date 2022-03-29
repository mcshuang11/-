import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def softmax(_outputs):
    maxes = np.max(_outputs, axis=-1, keepdims=True)
    shifted_exp = np.exp(_outputs - maxes)
    return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)


class TextClassificationPipeline:
    def __init__(self,
                 tokenizer,
                 model,
                 pad_to_max_length: bool = False,
                 max_seq_length=128):
        self.tokenizer = tokenizer
        self.model = model
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length
        self.per_device_eval_batch_size = 8

    def preprocess_function(self, examples):
        tokenized_inputs = self.tokenizer(
            examples,
            padding="max_length" if self.pad_to_max_length else False,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=True
        )
        return tokenized_inputs

    def __call__(self, sentences, *args, **kwargs):
        processed_dataset = self.preprocess_function(sentences)
        inputs = {
            "input_ids": tf.ragged.constant(processed_dataset["input_ids"]).to_tensor(),
            "attention_mask": tf.ragged.constant(processed_dataset["attention_mask"]).to_tensor(),
        }
        outputs = self.model.predict(inputs, batch_size=self.per_device_eval_batch_size)["logits"]
        scores_arr = softmax(outputs)
        res = []
        for scores in scores_arr:
            score_dic = {self.model.config.id2label[i]: score.item() for i, score in enumerate(scores)}
            sorted_score_items = sorted(score_dic.items(), key=lambda x: x[1], reverse=True)
            res.append([{"label": item[0], "score": item[1]} for item in sorted_score_items])
        return res


if __name__ == "__main__":
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        TFAutoModelForSequenceClassification,
    )

    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
    config = AutoConfig.from_pretrained("/mnt/nfs/wei.cai/bert4tf/hupu/p610786")
    model = TFAutoModelForSequenceClassification.from_pretrained("/mnt/nfs/wei.cai/bert4tf/hupu/p610786")
    pipe = TextClassificationPipeline(tokenizer=tokenizer, model=model)
    texts = ["兄弟 求一个威少群"]
    res = pipe(texts)
    print(res)
