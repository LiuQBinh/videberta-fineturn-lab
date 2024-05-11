# Use a pipeline as a high-level helper
from transformers import pipeline
pipe = pipeline("question-answering", model="Fsoft-AIC/videberta-xsmall")
print(pipe(question="Tôi sống ở đâu?", context="Tên tôi là Wolfgang và tôi sống ở Berlin"))



