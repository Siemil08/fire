import tensorflow as tf
import pandas as pd
import pymysql
from sklearn.preprocessing import RobustScaler
import numpy as np
import time

# tflite 모델 로드 및 인터프리터 생성
interpreter = tf.lite.Interpreter(model_path='fire_anomaly_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# DB 연결 함수 (기존 코드 유지)
def dbconnect():
    conn = pymysql.connect(
        host='localhost',
        port=3306,
        user='test1',
        password='P@ssw0rd',
        db='test',
        charset='utf8'
    )
    return conn

# 데이터 로드 및 전처리 함수 (기존 코드 유지)
def fetch_data(conn):
    query = "SELECT * FROM sensor_data"
    df = pd.read_sql(query, conn)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    features = ['CO_Room', 'Temperature_Room', 'Humidity_Room', 'PM10_Room']
    df = df.dropna(subset=features)
    df['CO_Room'] = df['CO_Room'].clip(lower=0)

    return df, features

# 시퀀스 생성 함수 (기존 코드 유지)
def create_sequences(df, features, window_size=60):
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])
    X_seq = []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i:i + window_size])
    return np.array(X_seq)

# DB에서 전체 데이터 불러오기
conn = dbconnect()
df, features = fetch_data(conn)
conn.close()

# 시퀀스 생성
X_seq = create_sequences(df, features)

print(f"총 시퀀스 개수: {len(X_seq)}\n시뮬레이션 시작...")

for i in range(len(X_seq)):
    input_seq = X_seq[i].astype(np.float32).reshape(input_details[0]['shape'])  # (1, 60, 4)
    
    interpreter.set_tensor(input_details[0]['index'], input_seq)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    prob = float(output_data[0][0])
    label = int(prob > 0.8)
    print(f"[{i+1}/{len(X_seq)}] 🔥 화재 확률: {prob:.4f} → {'🚨 경고' if label == 1 else '✅ 정상'}")

    if label == 1:
        print("\n🚨 경고 발생 시점의 원본 데이터 (최근 60개):")
        window_df = df.iloc[i:i+60][['Date'] + features]
        print(window_df.tail(60))  # 60개

    time.sleep(1)  # 1초 간격
