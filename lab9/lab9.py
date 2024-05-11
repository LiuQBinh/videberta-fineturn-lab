from datasets import load_dataset
squad = load_dataset("vile99/vi_quad")
print(squad['train'].column_names)


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


base_model_name="lqbin/videberta-xsmall_batch24_epoch30v6"
tokenizer = AutoTokenizer.from_pretrained('/mnt/c/Users/luuqu/Documents/GitHub/videberta-fineturn-lab/Fsoft-AIC/videberta-xsmall_batchsize24_epoch30')
model = AutoModelForQuestionAnswering.from_pretrained('/mnt/c/Users/luuqu/Documents/GitHub/videberta-fineturn-lab/Fsoft-AIC/videberta-xsmall_batchsize24_epoch30')

tokenizer.push_to_hub(base_model_name)
model.push_to_hub(base_model_name)