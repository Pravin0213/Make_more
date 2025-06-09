import torch 
import torch.nn.functional as F

names =  open('names.txt', 'r').read().splitlines()

all_chars =  ['.'] + sorted(list(set(''.join(names))))  
str_to_index = {ch: i for i, ch in enumerate(all_chars)}
index_to_str = {i: ch for i, ch in enumerate(all_chars)}

def data_set(names):
    block_size = 3  
    X, Y = [], []
    for name in names:
        context = [0]*block_size
        for char in name + '.':  # Append '.' to mark the end of the name
            ix = str_to_index[char]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # Shift the context window
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

n1 = int(0.8*len(names))  # 80% for training
n2 = int(0.8*len(names))  # 10% for validation

Xtrain, Ytrain = data_set(names[:n1])  # Training set
Xval, Yval = data_set(names[n2:])  # Validation set

embed_size = 10  # Size of character embeddings
block_size = 3  # Size of the context window

C = torch.randn((27,embed_size))  # Randomly initialize the character embeddings
W1 = torch.randn((embed_size*block_size,100))  # Randomly initialize the first layer weights. since we chose block size of 3, we have 3*2=6 input features (2 for each character in the context)
b1 = torch.randn(100)  # Randomly initialize the first layer biases
W2 = torch.randn((100,27))  # Randomly initialize the second layer weights
b2 = torch.randn(27)  # Randomly initialize the second layer biases

parameters = [C, W1, b1, W2, b2]

for param in parameters:
    param.requires_grad = True  # Enable gradient computation for all parameters
 
for lr in range(100000):
    # Forward pass
    ix = torch.randint(0, len(Xtrain), (64,))  # Randomly sample 32 indices
    emb = C[Xtrain[ix]]  # This size would be (N, block_size, 2) where N is the number of samples
    h = emb.view(-1 ,embed_size*block_size) @ W1 + b1  # Linear transformation
    activataion_1 = h.tanh()  # Activation function
    logits = activataion_1 @ W2 + b2  # Final linear transformation
    probs =logits.softmax(dim=1)  # Convert logits to probabilities
    loss = F.cross_entropy(logits, Ytrain[ix])  
    # Backward pass and update parameters
    for param in parameters:
        param.grad = None  # Zero out gradients before backward pass
    loss.backward()  # Backpropagation to compute gradients
    lr = 0.1
    if lr > 80000:
        lr = 0.01  # Learning rate
    for param in parameters:
        param.data += -lr * param.grad  # Update parameters with a learning rate of -50
print(loss.item())


emb = C[Xval]  # This size would be (N, block_size, 2) where N is the number of samples
h = emb.view(-1 ,embed_size*block_size) @ W1 + b1  # Linear transformation
activataion_1 = h.tanh()  # Activation function
logits = activataion_1 @ W2 + b2  # Final linear transformation
probs =logits.softmax(dim=1)  # Convert logits to probabilities
loss = F.cross_entropy(logits, Yval)
print(loss.item())


for _ in range(20):
    name_start = [0]*block_size
    out = []
    while True:
        emb = C[torch.tensor([name_start])]
        emb = torch.tanh(emb.view(1,-1)@W1 + b1)
        logits = emb @ W2 + b2
        probs = logits.softmax(dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        name_start = name_start[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(index_to_str[i] for i in out))