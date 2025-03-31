import requests
import pandas as pd
import time

# Binance API 엔드포인트
BASE_URL = "https://api.binance.com/api/v3/klines"

# XRP/USDT 5분봉 설정 ETHUSDT, BTCUSDT, XRPUSDT
symbol = "XRPUSDT"
interval = "5m"
limit = 1000  # 한 번에 가져올 최대 개수

# 현재 시간 (밀리초)
end_time = int(time.time() * 1000)

# 1년 전 시간 (밀리초)
three_years_ms = 365 * 24 * 60 * 60 * 1000
start_time = end_time - three_years_ms

# 데이터 저장용 리스트
all_data = []

while start_time < end_time:
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if not data:
        break  # 더 이상 데이터가 없으면 종료

    all_data.extend(data)
    
    # 가장 마지막 캔들의 타임스탬프를 다음 startTime으로 설정
    start_time = data[-1][0] + 1  
    time.sleep(0.5)  # API 요청 제한 방지

# Pandas DataFrame으로 변환
columns = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
           "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
df = pd.DataFrame(all_data, columns=columns)

# timestamp를 datetime 형식으로 변환
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# 필요한 컬럼만 선택
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

# CSV로 저장
df.to_csv("XRP_5min_1year_data.csv", index=False)
print("데이터 수집 완료, CSV 파일 저장 완료!")
