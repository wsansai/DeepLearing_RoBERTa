import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

# 读取CSV文件
df = pd.read_csv('textPreprocessing.csv')

# ================== 修改点1：三分类标签处理 ==================
# 数据预处理（保持与RoBERTa一致的标签映射）
texts = df['text'].values
labels = df['sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0}).values  # 修改为0/1/2

# ================== 修改点2：数据集7:1:2划分 ==================
# 第一次划分：分出测试集20%
X_temp, X_test, y_temp, y_test = train_test_split(
    texts, labels,
    test_size=0.2,
    stratify=labels,  # 保持类别分布
    random_state=42
)

# 第二次划分：从剩余数据中分出验证集10%（占原始数据的10%）
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.125,  # 0.125 * 0.8 = 0.1
    stratify=y_temp,  # 保持类别分布
    random_state=42
)

# ================== 修改点3：适配新划分的数据预处理 ==================
# 分词和填充序列（仅在训练集上拟合分词器）
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)  # 关键修改：只在训练集拟合

# 分别转换所有数据集
train_sequences = tokenizer.texts_to_sequences(X_train)
val_sequences = tokenizer.texts_to_sequences(X_val)
test_sequences = tokenizer.texts_to_sequences(X_test)

max_len = 100
X_train_pad = pad_sequences(train_sequences, maxlen=max_len)
X_val_pad = pad_sequences(val_sequences, maxlen=max_len)
X_test_pad = pad_sequences(test_sequences, maxlen=max_len)

# ================== 模型构建保持不变 ==================
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=max_len))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

# 编译模型（保持原优化器）
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ================== 修改点4：使用验证集监控训练 ==================
history = model.fit(
    X_train_pad, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val_pad, y_val),  # 使用验证集
    verbose=1
)

# ================== 可视化部分保持功能但适配新数据 ==================
# 绘制训练曲线
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

# ================== 测试集评估修正 ==================
# 测试集结果可视化（使用正确的测试集数据）
y_pred_prob = model.predict(X_test_pad[:100])  # 使用X_test_pad
y_pred = (y_pred_prob > 0.5).astype(int)

# 创建结果表格（使用X_test原始文本）
results = pd.DataFrame({
    'Text': X_test[:100],  # 使用划分后的测试集文本
    'True Label': y_test[:100],
    'Predicted Label': y_pred.flatten()
})
print("\n测试集前100条预测结果：")
print(results)