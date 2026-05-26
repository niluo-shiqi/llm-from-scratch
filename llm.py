import urllib.request
import os
import torch
from model import TinyLLM

char_to_idx={}
idx_to_char={}

def main():
    text = load_shakespeare()
    build_vocab(text)
    tokens=encode(text)
    print(f"Encoded {len(text)} characters into {len(tokens)} tokens")
    print(f"Vocab size: {len(char_to_idx)}")

    # create dataset
    sequence_length=256
    contexts, targets = create_dataset(tokens, sequence_length)
    print(f"Created {len(contexts)} training examples")

    # test the model
    model = TinyLLM(vocab_size=len(char_to_idx), embedding_dim=64, num_layers=4)

    # take first context(256 tokens)
    sample_input=torch.tensor(contexts[0]) # convert to tensor
    logits = model(sample_input)

    print(f"Input shape: {sample_input.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: [256, 65]")

    train(model, contexts[:1000], targets[:1000], num_epochs=3)



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

# return array of index numbers matched to given chars
def encode(text_sample):
    return [char_to_idx[c] for c in text_sample]

# create string of corresponding chars given array of index
def decode(tokens):
    return ''.join([idx_to_char[i] for i in tokens])

def create_dataset(tokens, seq_len):
    contexts=[]
    targets=[]

    for i in range(len(tokens) - seq_len):
        context = tokens[i : i+seq_len]
        target = tokens[i+seq_len]
        contexts.append(context)
        targets.append(target)
    return contexts, targets

def train(model, contexts, targets, num_epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # create optimizer
    loss_fn = torch.nn.CrossEntropyLoss() # create loss function

    for epoch in range(num_epochs):
        for step, (context, target) in enumerate(zip(contexts, targets)):

            # forward pass
            context = torch.tensor(context)
            target = torch.tensor(target)
            logits = model(context)
            last_logit = logits[-1]

            #compute loss
            loss = loss_fn(last_logit, target)
            
            # clear for next step
            optimizer.zero_grad() 

            # backward pass
            loss.backward()

            # update weights
            optimizer.step() 

            #print progress every 100 steps
            if (step % 100) == 0:
                print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")




if __name__ == "__main__":
    main()