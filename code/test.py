from transformers import BertTokenizer, BertForSequenceClassification


# 指定缓存目录为当前目录下的文件夹
model_path = "./model"

# 加载分词器和模型，指定分类标签数为2（正面/负面）
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=model_path)
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2, cache_dir=model_path)