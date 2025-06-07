import torch 
import torch.nn.functional as F

names =  open('names.txt', 'r').read().splitlines()

all_chars =  ['.'] + sorted(list(set(''.join(names))))  
str_to_index = {ch: i for i, ch in enumerate(all_chars)}

index_to_str = {i: ch for i, ch in enumerate(all_chars)}


xs , ys = [], []
for name in names: 
    chs = ['.'] + list(name) + ['.'] # We do this to add <S> to first token and <E> to last token as to make a pair in starting and ending of the name
    for char1, char2 in zip(chs, chs[1:]):
        i1 = str_to_index[char1]
        i2 = str_to_index[char2]
        xs.append(i1)
        ys.append(i2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
W= torch.randn((27,27), requires_grad=True)

for i in range(100):
    xenc = F.one_hot(xs, num_classes=len(all_chars)).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)  # Normalize the rows to get probabilities
    loss = probs[torch.arange(len(ys)), ys].log().mean()*(-1)  # Log probabilities of the true labels
    print(loss.item())
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad 

for i in range(10): # This will print 10 samples
    out = []
    ix = 0 # Start with the first character
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=len(all_chars)).float()
        logits = xenc @ W
        P = logits.exp() / logits.exp().sum()
        ix = torch.multinomial(P, num_samples=1, replacement=True).item() # Sample the next character index
        out.append(index_to_str[ix])
        if ix == 0: # If we hit the end character, break the loop
            break
    print(''.join(out))