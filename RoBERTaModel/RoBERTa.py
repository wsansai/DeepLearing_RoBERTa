import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, create_optimizer
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 加载数据
df = pd.read_csv('textPreprocessing.csv')  # 确保文件路径正确
texts = df['text'].values
labels = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0}).values

# 2. 划分数据集（训练集 80%、测试集 20%）
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels,
    test_size=0.2,
    random_state=42  # 固定随机种子保证可复现
)

# 3. 从训练集中再划分验证集（10% 原始数据，即训练集的 12.5%）
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels,
    test_size=0.125,  # 0.125 * 0.8 = 0.1（占原始数据 10%）
    random_state=42
)

# 4. 数据预处理：使用 RoBERTa 分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_data(texts):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='tf'
    )

# 对训练集、验证集、测试集分别分词
train_encodings = tokenize_data(train_texts)
val_encodings = tokenize_data(val_texts)
test_encodings = tokenize_data(test_texts)

# 5. 创建 TensorFlow Dataset
def create_dataset(input_ids, attention_mask, labels, batch_size=64):
    return tf.data.Dataset.from_tensor_slices((
        {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        },
        labels
    )).batch(batch_size)

# 创建训练集、验证集、测试集 Dataset
train_dataset = create_dataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
val_dataset = create_dataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)
test_dataset = create_dataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

# 6. 构建 RoBERTa 模型
model = TFRobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=3  # 三分类任务（positive/neutral/negative）
)

# 7. 定义优化器（含学习率预热和权重衰减）
epochs = 10
total_train_steps = len(train_dataset) * epochs  # 总训练步数
optimizer, _ = create_optimizer(
    init_lr=2e-5,              # 初始学习率（推荐范围 2e-5 ~ 5e-5）
    num_train_steps=total_train_steps,
    weight_decay_rate=6e-3,    # 权重衰减防止过拟合
    num_warmup_steps=int(0.1 * total_train_steps)  # 前 10% 步数用于预热
)

# 8. 编译模型
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 9. 训练模型（使用验证集监控）
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset  # 使用独立的验证集
)

# 10. 绘制训练曲线
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

# 11. 在独立测试集上评估模型性能
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test set loss: {test_loss:.4f}")
print(f"Test set accuracy: {test_accuracy:.4f}")

# 12. 预测测试集并保存结果
predictions = model.predict(test_dataset)
predicted_labels = tf.argmax(predictions.logits, axis=1).numpy()

results = pd.DataFrame({
    'Text': test_texts,
    'True Label': test_labels,
    'Predicted Label': predicted_labels
})

# 将预测结果保存为 CSV
results.to_csv('prediction_results.csv', index=False)
print("The result is saved as 'prediction_results.csv'")