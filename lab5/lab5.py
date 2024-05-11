from datasets import load_dataset
squad = load_dataset("vile99/vi_quad")
print(squad['train'].column_names)


from transformers import AutoTokenizer, DebertaV2ForQuestionAnswering
import torch


base_model_name="Fsoft-AIC/videberta-xsmall"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = DebertaV2ForQuestionAnswering.from_pretrained(base_model_name)

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]

        if(answer is None):
            start_positions.append(0)
            end_positions.append(0)
            continue

        # start_char = answer["answer_start"][0]
        # end_char = answer["answer_start"][0] + len(answer["text"][0])

        start_char = answer["answer_start"]
        end_char = answer["answer_start"] + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)

from transformers import DefaultDataCollator
data_collator = DefaultDataCollator()



###################################
bactch_size=24
num_train_epochs=30
model_save_name=base_model_name + '_batch' + str(bactch_size) + '_epoch' + str(num_train_epochs)
###################################
from transformers import TrainingArguments, Trainer
args = TrainingArguments(
    output_dir=model_save_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=bactch_size,
    per_device_eval_batch_size=bactch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    push_to_hub=True,
)
model.resize_token_embeddings(len(tokenizer))
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_squad["train"],
    eval_dataset=tokenized_squad["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()