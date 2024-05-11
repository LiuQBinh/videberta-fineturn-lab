from transformers import AutoTokenizer, DebertaV2ForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("/mnt/c/Users/luuqu/Documents/GitHub/videberta-fineturn-lab/Fsoft-AIC/videberta-xsmall_batch24_epoch10")
model = DebertaV2ForQuestionAnswering.from_pretrained("/mnt/c/Users/luuqu/Documents/GitHub/videberta-fineturn-lab/Fsoft-AIC/videberta-xsmall_batch24_epoch10")

question, text = "Tôi sống ở đâu?", "Tên tôi là Wolfgang và tôi sống ở Berlin"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print('-----------------------------------------------------------------------')
print(tokenizer.decode(predict_answer_tokens))


# target is "nice puppet"
# target_start_index = torch.tensor([17])
# target_end_index = torch.tensor([17])
# print('tokenizer.decode(target_answer_tokens)')
# print('tokenizer.decode(target_answer_tokens)')
# print('tokenizer.decode(target_answer_tokens)')
# print(tokenizer.decode(inputs.input_ids[0, target_start_index : target_end_index + 1]))

# for index, token in enumerate(inputs.input_ids.numpy()[0]):
#     print('token %s : %s' % (index, tokenizer.decode(token)))


# outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
# loss = outputs.loss