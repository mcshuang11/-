import pandas as pd
import torch


class TapasQuestioningAnswerPipeline:
    def __init__(self,
                 tokenizer,
                 model):
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        if self.device == "cuda":
            self.model.to(self.device)
        self.model.eval()

    def __call__(self, table, queries, *args, **kwargs):
        inputs = self.tokenizer(table=table, queries=queries, padding='max_length', return_tensors="pt", truncation=True)

        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        inputs = {k: v.cpu() for k, v in inputs.items()}

        # extract answer coordinates and aggregation type index
        predicted_answer_coordinates, predicted_aggregation_indices = self.tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits.cpu().detach(),
            outputs.logits_aggregation.cpu().detach()
        )

        id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
        aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

        # converge answers and answer cell coordinates
        answers = []
        for coordinates in predicted_answer_coordinates:
            answer = []
            if len(coordinates) == 1:
                # only a single cell:
                # add answers and then coordinates
                answer.append([table.iat[coordinates[0]]])
                answer.append(coordinates)
            else:
                # multiple cells
                cell_values = []
                for coordinate in coordinates:
                    cell_values.append(table.iat[coordinate])
                answer.append(cell_values)
                answer.append(coordinates)
            answers.append(answer)

        # output to a dict
        result = []
        for query, answer, predicted_agg in zip(queries, answers, aggregation_predictions_string):
            result.append({'answer': answer[0],
                           'coordinates': answer[1],
                           'aggregation': predicted_agg})
        return result


if __name__ == "__main__":
    from transformers import TapasTokenizer, TapasForQuestionAnswering
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base-finetuned-wtq")
    model = TapasForQuestionAnswering.from_pretrained("google/tapas-base-finetuned-wtq")
    pipe = TapasQuestioningAnswerPipeline(tokenizer=tokenizer, model=model)

    data = {'Cities': ["Paris, France", "London, England", "Lyon, France"], 'Inhabitants': ["2.161", "8.982", "0.513"]}
    queries = ["Which city has most inhabitants?", "How many French cities are in the list?",
               "How many inhabitants live in French cities?"]
    table = pd.DataFrame.from_dict(data)

    res = pipe(table, queries)
    print(res)
    # import ipdb;ipdb.set_trace()
