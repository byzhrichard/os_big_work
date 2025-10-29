from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset


download_model_path = "./model"
download_data_path = "./data"
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=download_model_path)
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2, cache_dir=download_model_path)
dataset = load_dataset('lansinuote/ChnSentiCorp', cache_dir=download_data_path)
exit()

# # 加载分词器和模型，指定分类标签数为2（正面/负面）
# model_path = "./model/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
#
# # 加载数据集
# dataset = load_dataset('./data/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a')
