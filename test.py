from transformers import BertTokenizer
import torch
import uvicorn
from fastapi import FastAPI
from bert import bert_ATE, bert_ABSA
import json
from fastapi.middleware.cors import CORSMiddleware
#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device( "cpu")
pretrain_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(pretrain_model_name)
model_ATE = bert_ATE(pretrain_model_name).to(DEVICE)
model_ABSA = bert_ABSA(pretrain_model_name).to(DEVICE)
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_model(model, path):
    model.load_state_dict(torch.load(path), strict=False)
    return model

model_ABSA = load_model(model_ABSA, 'bert_ABSA2.pkl')
model_ATE = load_model(model_ATE, 'bert_ATE.pkl')

def predict_model_ABSA(sentence, aspect, tokenizer,return_dict=False):
    t1 = tokenizer.tokenize(sentence)
    t2 = tokenizer.tokenize(aspect)

    word_pieces = ['[cls]']
    word_pieces += t1
    word_pieces += ['[sep]']
    word_pieces += t2

    segment_tensor = [0] + [0]*len(t1) + [0] + [1]*len(t2)

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)
    segment_tensor = torch.tensor(segment_tensor).to(DEVICE)

    with torch.no_grad():
        outputs = model_ABSA(input_tensor, None, None, segments_tensors=segment_tensor,return_dict=False)
        _, predictions = torch.max(outputs, dim=1)
    
    return word_pieces, predictions, outputs

def predict_model_ATE(sentence, tokenizer,return_dict=False):
    word_pieces = []
    tokens = tokenizer.tokenize(sentence)
    word_pieces += tokens

    ids = tokenizer.convert_tokens_to_ids(word_pieces)
    input_tensor = torch.tensor([ids]).to(DEVICE)

    with torch.no_grad():
        outputs = model_ATE(input_tensor, None, None,return_dict=False)
        _, predictions = torch.max(outputs, dim=2)
    predictions = predictions[0].tolist()

    return word_pieces, predictions, outputs

@app.get('/')
def index():
    return {'message': 'Hello, World'}
@app.get('/MenuCritic')
def get_name():
    return {'Welcome To MenuCritic': 'MenuCritic'}
@app.post('/predict')
def ATE_ABSA(text:str):
    terms = []
    word = ""
    x, y, z = predict_model_ATE(text, tokenizer, return_dict=False)
    for i in range(len(y)):
        if y[i] == 1:
            if len(word) != 0:
                terms.append(word)
            word = x[i]
        if y[i] == 2:
            word += (" " + x[i])
            
    
    if len(word) != 0:
            terms.append(word)
            
    combined_terms = []
    #i = 0
    #while i < len(terms):
    #    if "##" in terms[i]:
    #        combined_term = terms[i].replace("##", "")
    #        j = i + 1
    #        while j < len(terms) and "##" in terms[j]:
    #            combined_term += terms[j].replace("##", "")
    #            j += 1
    #        combined_terms.append(combined_term)
    #        i = j
    #    else:
    #        combined_terms.append(terms[i])
    #        i += 1
    index=0
    print(terms)
    for i in range (0,len(terms)):
        combined_term=""
        flag=False
        if "##" in terms[i]:
            original_i = i-1
            while "##" in terms[i] and i+1!=len(terms):
                temp_str = terms[i]
                temp_term = temp_str.replace("##","")
                combined_term += temp_term
                i += 1
            combined_terms.append(terms[original_i]+combined_term)
            
            i+=1
        elif((i!=len(terms)-1 ) and ("##" not in terms[i+1])):
            combined_terms.append(terms[i])
        elif(i==len(terms)-1 and "##" not in terms[i]):
            combined_terms.append(terms[i])
    index=0
    combined_terms_copy = combined_terms.copy()
    for term in combined_terms_copy:
                    if "##" in term:
                        combined_terms.remove(term)
    print(combined_terms)
    #i=0
    #while i < len(terms):
    #    if "##" in terms[i]:
    #        combined_term = terms[i].replace("##", "")
    #        j = i + 1
    #        while j < len(terms) and "##" in terms[j]:
    #            combined_term += terms[j].replace("##", "")
    #            j += 1
    #        combined_terms.append(combined_term)
    #        i = j
    #    else:
    #        combined_terms.append(terms[i])
    #        i += 1
           
            
    
    output_dict = {"output": []}
    if len(combined_terms) != 0:
        for term in combined_terms:
            term = term.replace("##", "")
            _, c, p = predict_model_ABSA(text, term, tokenizer)
            output_dict["output"].append({"term": term, "class": int(c)})
    print(json.dumps(output_dict))
    return output_dict


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
