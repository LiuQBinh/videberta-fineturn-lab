from datasets import load_dataset
squad = load_dataset("vile99/vi_quad")
print(squad['train'].column_names)


from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


base_model_name="/mnt/c/Users/luuqu/Documents/GitHub/videberta-fineturn-lab/lab10/Fsoft-AIC/videberta-xsmall_batchsize24_epoch1"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForQuestionAnswering.from_pretrained(base_model_name)

tokenizer.push_to_hub('videberta-xsmall_batchsize24_epoch1')
model.push_to_hub('videberta-xsmall_batchsize24_epoch1')