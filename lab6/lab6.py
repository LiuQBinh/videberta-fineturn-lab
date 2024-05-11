from transformers import AutoTokenizer, DebertaV2ForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("/mnt/c/Users/luuqu/Documents/GitHub/videberta-fineturn-lab/Fsoft-AIC/videberta-xsmall_batch24_epoch30/checkpoint-27500")
model = DebertaV2ForQuestionAnswering.from_pretrained("/mnt/c/Users/luuqu/Documents/GitHub/videberta-fineturn-lab/Fsoft-AIC/videberta-xsmall_batch24_epoch30/checkpoint-27500")

question, text = "Việc đưa ra các Chính sách đã tác động điều gì với Malaysia?", "Sắc tộc có ảnh hưởng lớn trong chính trị Malaysia, nhiều chính đảng dựa trên nền tảng dân tộc. Các hành động quả quyết như Chính sách Kinh tế mới và thay thế nó là Chính sách Phát triển Quốc gia, được thực hiện nhằm thúc đẩy địa vị của bumiputera, bao gồm người Mã Lai và các bộ lạc bản địa, trước những người phi bumiputera như người Malaysia gốc Hoa và người Malaysia gốc Ấn. Các chính sách này quy định ưu đãi cho bumiputera trong việc làm, giáo dục, học bổng, kinh doanh, tiếp cận nhà giá rẻ hơn và hỗ trợ tiết kiệm. Tuy nhiên, nó gây ra oán giận rất lớn giữa các dân tộc."

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
print('tokenizer.decode(predict_answer_tokens)')
print('tokenizer.decode(predict_answer_tokens)')
print('tokenizer.decode(predict_answer_tokens)')
print(tokenizer.decode(predict_answer_tokens))


# target is "nice puppet"
target_start_index = torch.tensor([17])
target_end_index = torch.tensor([17])
print('tokenizer.decode(target_answer_tokens)')
print('tokenizer.decode(target_answer_tokens)')
print('tokenizer.decode(target_answer_tokens)')
print(tokenizer.decode(inputs.input_ids[0, target_start_index : target_end_index + 1]))

for index, token in enumerate(inputs.input_ids.numpy()[0]):
    print('token %s : %s' % (index, tokenizer.decode(token)))


outputs = model(**inputs, start_positions=target_start_index, end_positions=target_end_index)
loss = outputs.loss