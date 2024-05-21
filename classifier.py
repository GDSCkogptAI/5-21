import pandas as pd

# CSV 파일 읽기
data = pd.read_csv('C:/kogpt2/result.csv')

# 필요한 열만 선택
data = data[['question', 'label']]


from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['question'])
y = data['label']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 데이터셋 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 로지스틱 회귀 모델 훈련
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 정확도 및 분류 보고서 출력
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import pickle

# 모델 및 벡터라이저 저장
with open('classifier_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

# 모델 및 벡터라이저 로드
with open('classifier_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    loaded_vectorizer = pickle.load(vec_file)

# 새로운 데이터 예측
new_questions = ["화이트데이에 고백할까요?", "DB하이텍의 2011년 12월 유동비율은 얼마야"]
new_X = loaded_vectorizer.transform(new_questions)
predictions = loaded_model.predict(new_X)
print(predictions)
