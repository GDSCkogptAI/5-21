from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
SENT = "<unused1>"

data_path = 'C:/kogpt2/ChatBotData.csv'
model_path = 'C:/kogpt2/chat_model'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('result.csv')

with open('C:/kogpt2/classifier/classifier_model.pkl', 'rb') as model_file:
    classifier_model = pickle.load(model_file)

with open('C:/kogpt2/classifier/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

tokenizer = PreTrainedTokenizerFast.from_pretrained("C:/kogpt2/tokenizer_with_custom_tokens", bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token=MASK)
chat_model = GPT2LMHeadModel.from_pretrained(model_path)
chat_model.to(device)

# 질문에 대한 응답을 제공하는 함수 정의
def get_response(question, chat_model, classifier_model, vectorizer, data, tokenizer, device):
    # 질문을 벡터화
    question_vector = vectorizer.transform([question])
    
    # 분류 모델로 예측
    prediction = classifier_model.predict(question_vector)[0]
    
    if prediction == 0:
        # 기존 챗봇 모델로 답변 생성
        input_text = Q_TKN + question + SENT + A_TKN
        chat_model.eval()
        with torch.no_grad():
            a = ""
            while True:
                input_ids = torch.LongTensor(tokenizer.encode(Q_TKN + question + SENT + A_TKN + a)).unsqueeze(dim=0).to(device)
                pred = chat_model(input_ids)
                pred = pred.logits
                gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
            response = a.strip()
    else:
        # result.csv에서 질문에 해당하는 답변 찾기
        answer = data[data['question'] == question]['answer']
        if not answer.empty:
            response = answer.values[0]
        else:
            response = "질문에 해당하는 답변을 찾을 수 없습니다."
    
    return response

# Flask API 엔드포인트 정의

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('question')
    if not user_input:
        return jsonify({'error': 'No question provided'}), 400
    
    response = get_response(user_input, chat_model, classifier_model, vectorizer, data, tokenizer, device)
    return jsonify({'response': response})

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



