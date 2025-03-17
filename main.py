from transformers import T5Model, T5ForConditionalGeneration, T5Tokenizer


t5_model = T5Model.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")