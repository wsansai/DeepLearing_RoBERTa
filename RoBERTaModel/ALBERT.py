import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AlbertTokenizer, TFAlbertForSequenceClassification, create_optimizer
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 加载数据
df = pd.read_csv('sentiment_analysis.csv')

# 提取 text 和 sentiment 列
texts = df['text'].values
labels = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0}).values

# 2. 统一数据集划分（7:1:2）
# 第一次划分：分出测试集20%
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 第二次划分：分出验证集10%
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels,
    test_size=0.125,  # 0.125 * 0.8 = 0.1
    stratify=train_labels,
    random_state=42
)

# 3. 数据预处理
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

def tokenize_data(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=128,  # 统一长度
        return_tensors='tf'
    )

# 分别处理各数据集
train_encodings = tokenize_data(train_texts)
val_encodings = tokenize_data(val_texts)
test_encodings = tokenize_data(test_texts)

# 4. 创建统一格式的Dataset
def create_dataset(input_ids, attention_mask, labels, batch_size=64):  # 统一批量大小
    return tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        },
        labels
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_dataset = create_dataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = create_dataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
test_dataset = create_dataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

# 5. 构建模型（保持相同分类头）
model = TFAlbertForSequenceClassification.from_pretrained(
    'albert-base-v2',
    num_labels=3  # 三分类任务
)

# 6. 统一优化器配置
epochs = 10  # 统一训练轮次
total_train_steps = len(train_dataset) * epochs
optimizer, _ = create_optimizer(
    init_lr=2e-5,
    num_train_steps=total_train_steps,
    weight_decay_rate=6e-3,  # 相同权重衰减
    num_warmup_steps=int(0.1 * total_train_steps)  # 相同预热策略
)

# 7. 编译模型（相同配置）
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 8. 训练模型（相同验证策略）
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,  # 使用独立验证集
    verbose=1
)

# 9. 统一可视化代码
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# 10. 统一评估流程
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")

# 11. 统一结果保存格式
predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

results = pd.DataFrame({
    'Text': test_texts,
    'True Label': test_labels,
    'Predicted Label': predicted_labels
})

results.to_csv('albert_prediction_results.csv', index=False)
print("ALBERT预测结果已保存")