# python3 -m pip install pandas numpy tensorflow scikit-learn matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

# 데이터셋 로드
data = pd.read_csv('BTC_5min_1year_data.csv')  # 파일명은 실제 파일에 맞게 수정

# 데이터 확인
print(data.head())

# 데이터 전처리 (Scaling)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume']])

# 시퀀스 생성 (30개의 5분봉 → 약 150분을 한 시퀀스로 설정)
def create_sequences(data, seq_length=30):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 30
sequences = create_sequences(data_scaled, seq_length)

# LSTM Autoencoder 모델 구성
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 5), return_sequences=False),
    RepeatVector(seq_length),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(5))
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 모델 학습 (정상적인 데이터로만 학습한다고 가정)
model.fit(sequences, sequences, epochs=50, batch_size=64, validation_split=0.1)

# ✅ 학습 완료 후 모델 저장
model.save('btc_5min_autoencoder.h5')
print("✅ 모델이 저장되었습니다: btc_5min_autoencoder.h5")

# 이상탐지 점수 계산 (복원 오차)
reconstructed = model.predict(sequences)
train_loss = np.mean(np.square(sequences - reconstructed), axis=(1, 2))

# 복원오차 시각화 (이상 거래는 복원 오차가 클 것으로 예상)
plt.figure(figsize=(12,6))
plt.plot(train_loss, label='Reconstruction Loss')
plt.xlabel('Time')
plt.ylabel('Loss (MSE)')
plt.title('LSTM Autoencoder Anomaly Detection')
plt.legend()
plt.show()

# 이상거래 임계값 설정 예시 (평균 + 3 표준편차)
thresh = np.mean(train_loss) + 3 * np.std(train_loss)
print(f'이상 탐지 임계값: {thresh}')

# 이상치 탐지된 인덱스 출력
anomalies = np.where(train_loss > thresh)[0]
print(f'이상 거래로 탐지된 데이터 인덱스: {anomalies}')
