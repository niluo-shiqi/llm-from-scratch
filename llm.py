import urllib.request
import os

def main():
    text = load_shakespeare()
    print(f"loaded {len(text)} characters")
    print("first 200 chars: ")
    print(text[:200])

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
        

if __name__ == "__main__":
    main()