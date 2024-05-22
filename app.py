from main import get_response
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