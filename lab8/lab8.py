from transformers import pipeline
pipe = pipeline("question-answering", model="lqbin/videberta-xsmall_batch24_epoch30")


question, context = "Mối quan hệ giữa sinh viên năm nhất và bằng tốt nghiệp THPT", "Em là sinh viên năm nhất mới nhận được bằng tốt nghiệp THPT"
print(pipe(question=question, context=context))
