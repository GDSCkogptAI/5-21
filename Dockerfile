# 베이스 이미지 설정
FROM python:3.8

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 파일 복사
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY main.py /app/main.py
COPY result.csv /app/result.csv
COPY classifier/classifier_model.pkl /app/classifier_model.pkl
COPY classifier/vectorizer.pkl /app/vectorizer.pkl
COPY chat_model /app/chat_model


# 필요한 패키지 설치
RUN pip install --no-cache-dir -r /app/requirements.txt

# Flask 서버 실행
CMD ["python", "/app/app.py"]
