import tensorflow as tf
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


class MultiTextClassificationPipeline:
    def __init__(self,
                 tokenizer,
                 model,
                 pad_to_max_length: bool = False,
                 max_seq_length=128):
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
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
        predictions = self.model.predict(inputs, batch_size=self.per_device_eval_batch_size)["logits"]
        predicted_class = np.where(predictions > 0.5, 1, 0)
        predicted_labels = []
        for label in predicted_class:
            for index, item in enumerate(label):
                if item == 1:
                    item = config.id2label[index]
                    predicted_labels.append(item)
        return predicted_labels


if __name__ == "__main__":
    from transformers import (
        AutoTokenizer,
        AutoConfig,
        TFAutoModelForSequenceClassification,
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    config = AutoConfig.from_pretrained("/home/shuang.mo/bert4tf/outputs/toxic")
    model = TFAutoModelForSequenceClassification.from_pretrained("/home/shuang.mo/bert4tf/outputs/toxic")
    pipe = MultiTextClassificationPipeline(tokenizer=tokenizer, model=model)
    texts = ["God fuck you too, pimple ass.  My additions are better, you guys are just to stupid to realize it."]
    res = pipe(texts)
    print(res)
