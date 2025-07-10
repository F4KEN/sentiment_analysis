import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

df = pd.read_csv("veri.csv")

# Clean text
def temizle(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zçğıöşü\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["yorum_temiz"] = df["yorum"].apply(temizle)

# Tokenization and padding
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(df["yorum_temiz"])
X = tokenizer.texts_to_sequences(df["yorum_temiz"])
X_pad = pad_sequences(X, maxlen=20)
y = df["etiket"].values

model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=20),
    LSTM(64, dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_pad, y, epochs=10, batch_size=2, validation_split=0.2)

# Prediction function
def predict_yorum(comment):
    cleaned = temizle(comment)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=20)
    prediction = model.predict(pad, verbose=0)[0][0]
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'
    return f"'{comment}' → {sentiment} ({prediction:.2f})"

# Sample predictions
print(predict_yorum("Bu video çok güzel olmuş"))
print(predict_yorum("Nefret ettim, kötüydü"))
print(predict_yorum("Harika anlatım hocam"))
print(predict_yorum("Harika."))
