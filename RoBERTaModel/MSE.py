import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification, create_optimizer
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 加载数据并预处理
df = pd.read_csv('textPreprocessing.csv')
texts = df['text'].values
labels = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0}).values

# 划分数据集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.125, random_state=42)

# 标签转换为 one-hot 编码
train_labels_onehot = tf.keras.utils.to_categorical(train_labels, num_classes=3)
val_labels_onehot = tf.keras.utils.to_categorical(val_labels, num_classes=3)
test_labels_onehot = tf.keras.utils.to_categorical(test_labels, num_classes=3)

# 2. 数据预处理
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
def tokenize_data(texts):
    return tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors='tf')

train_encodings = tokenize_data(train_texts)
val_encodings = tokenize_data(val_texts)
test_encodings = tokenize_data(test_texts)

# 3. 创建 Dataset
def create_dataset(input_ids, attention_mask, labels, batch_size=64):
    return tf.data.Dataset.from_tensor_slices(({'input_ids': input_ids, 'attention_mask': attention_mask}, labels)).batch(batch_size)

train_dataset = create_dataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels_onehot)
val_dataset = create_dataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels_onehot)
test_dataset = create_dataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels_onehot)

# 4. 构建模型
model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# 5. 定义优化器（使用 MSE 作为损失函数）
epochs = 10
total_train_steps = len(train_dataset) * epochs
optimizer, _ = create_optimizer(
    init_lr=2e-5,
    num_train_steps=total_train_steps,
    weight_decay_rate=5e-3,
    num_warmup_steps=int(0.1 * total_train_steps)
)

# 编译模型（关键修改）
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.MeanSquaredError(),  # 使用 MSE
    metrics=['accuracy']
)

# 6. 训练模型
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
)
# 10. 绘制训练曲线（完整代码）
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 11. 在独立测试集上评估模型性能（完整代码）
print("\n--- 测试集最终评估 ---")
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")


