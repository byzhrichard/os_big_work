from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# 加载分词器和模型，指定分类标签数为2（正面/负面）
model_path = "./model/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# 加载数据集
dataset = load_dataset('./data/lansinuote___chn_senti_corp/default/0.0.0/b0c4c119c3fb33b8e735969202ef9ad13d717e5a')

# 使用分词器预处理数据
def tokenize_function(examples):
    # 分词器会自动进行分词、截断和填充
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=128)

encoded_dataset = dataset.map(tokenize_function, batched=True)
# 设置格式为PyTorch张量
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])



################

from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score

# 定义评估指标计算函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc}

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮数
    per_device_train_batch_size=8,   # 训练批次大小
    per_device_eval_batch_size=8,    # 评估批次大小
    eval_strategy="epoch",           # 每个epoch后评估
    save_strategy="epoch",           # 每个epoch后保存模型
    logging_dir='./logs',            # 日志目录
    learning_rate=2e-5,              # 学习率
    load_best_model_at_end=True,     # 训练结束后加载最佳模型
)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],  # 或 "test"
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 保存微调后的模型与分词器
model.save_pretrained('./my_sentiment_model')
tokenizer.save_pretrained('./my_sentiment_model')