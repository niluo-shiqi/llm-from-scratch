import urllib.request
import os

char_to_idx={}
idx_to_char={}

def main():
    text = load_shakespeare()
    build_vocab(text)

    tokens=encode(text)
    print(f"Encoded {len(text)} characters into {len(tokens)} tokens")
    print(f"Vocab size: {len(char_to_idx)}")


# load shakespeare dataset
def load_shakespeare():
    txt_path="data/shakespeare.txt"
    if not os.path.exists(txt_path):
        os.makedirs("data", exist_ok=True)
        response = urllib.request.urlopen(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )
        text = response.read().decode('utf-8')
    
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
            return text
        
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
        return text

# tokenization

# get all unique characters from text, sort them, create dictionary from each char to an index and vice versa, print size of vocab
def build_vocab(text):
    unique_chars= sorted(set(text))
    index=0
    for item in unique_chars:
        char_to_idx.update({item: index})
        idx_to_char.update({index: item})
        index+=1

def encode(text_sample):
    return [char_to_idx[c] for c in text_sample]

def decode(tokens):
    return ''.join([idx_to_char[i] for i in tokens])
    
        

if __name__ == "__main__":
    main()