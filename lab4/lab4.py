from datasets import load_dataset
squad = load_dataset("vile99/vi_quad")
print(squad['train'][0])