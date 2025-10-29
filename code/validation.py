from transformers import BertTokenizer, BertForSequenceClassification
import torch
from datasets import load_dataset

# =======================
# 1. 加载模型和分词器
# =======================
model_path = './my_sentiment_model'  # 训练后保存的路径
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 如果有GPU可用则使用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# =======================
# 2. 定义预测函数
# =======================
def predict_sentiment(texts):
    """
    输入: texts -> str 或 list[str]
    输出: 每个文本的情感标签 (0=负面, 1=正面)
    """
    if isinstance(texts, str):
        texts = [texts]

    # 分词并转为tensor
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

    return predictions.cpu().numpy()


# =======================
# 3. 单条文本测试
# =======================
examples = [
    "这个电影真的太好看了！剧情感人，演员演技出色。",
    "这家餐厅的服务太差了，再也不会去了。"
]

preds = predict_sentiment(examples)
for text, pred in zip(examples, preds):
    label = "正面" if pred == 1 else "负面"
    print(f"【{label}】 {text}")

# =======================
# 4. 在测试集上评估准确率（修正版）
# =======================

dataset = load_dataset('./data/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a')
test_dataset = dataset["test"]


# 分词
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


encoded_test = test_dataset.map(tokenize_function, batched=True)
encoded_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

correct = 0
total = 0

for batch in torch.utils.data.DataLoader(encoded_test, batch_size=8):
    # 分离输入和标签
    labels = batch["label"].to(device)
    inputs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device)
    }

    # 前向传播
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)

    correct += (preds == labels).sum().item()
    total += len(labels)

print(f"测试集准确率: {correct / total:.4f}")