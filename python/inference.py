import torch
import json
import spacy
from model import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Function to load vocabulary from a .json file
def load_vocab_from_json(vocab_path, max_size):
    vocab = {}
    try:
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    except FileNotFoundError:
        print(f"{vocab_path} not found.")
        return {}
    
    #Filtering out indices > max_size - 1
    vocab = {k: v for k, v in vocab.items() if v < max_size}    
    return vocab

nlp = spacy.load('pl_core_news_sm')

def inference(src_text, src_vocab, tgt_vocab, model_path='model.pt', max_len=10):
    input_vocab_size = 56244
    output_vocab_size = 60135

    #Loading model
    model = create_model(input=input_vocab_size, output=output_vocab_size, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #Source text preprocessing
    doc = nlp(src_text)
    src_indices = [src_vocab.get(token.text.lower(), src_vocab.get('<UNK>', 0)) for token in doc]
    if None in src_indices:
        print(f"Error: Some tokens in '{src_text}' mapped to None: {src_indices}")
        return
    src_indices = [min(max(idx, 0), input_vocab_size - 1) for idx in src_indices]
    src = torch.tensor([src_indices], device=device)

    #Start/end tokens
    start_token_idx = tgt_vocab['<SOS>']
    end_token_idx = tgt_vocab['<EOS>']


    print(f"Source indices: {src_indices}")

    #Inference
    with torch.no_grad():
        output_sequences = model.infer(src, start_token_idx, end_token_idx, max_len=max_len)

    #Converting to tokens
    for seq in output_sequences:
        tokens = [word for word,idx in tgt_vocab.items() if idx in seq and word not in ['<SOS>', '<EOS>', '<UNK>']]
        sentence = ' '.join(tokens)
        print(f"Translated sentence: {sentence}")

if __name__ == "__main__":
    #Loading vocabularies with size caps
    src_vocab = load_vocab_from_json('../data/src_vocab.json', max_size=56244)
    tgt_vocab = load_vocab_from_json('../data/tgt_vocab.json', max_size=60135)

    #Checking vocabulary max indexes
    print(f"Max source index: {max(src_vocab.values(), default=-2)}")
    print(f"Max target index: {max(tgt_vocab.values(), default=-2)}")

    #Inference
    input_text = "Dzień dobry" #Example input
    normalized_input = input_text.lower()
    inference(normalized_input, src_vocab, tgt_vocab, model_path='../model_seq2seq.pt', max_len=10)