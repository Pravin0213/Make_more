import torch
names =  open('names.txt', 'r').read().splitlines()


b= {}

for name in names: 
    chs = ['<S>'] + list(name) + ['<E>'] # We do this to add <S> to first token and <E> to last token as to make a pair in starting and ending of the name
    for char1, char2 in zip(chs, chs[1:]):
        bigram = (char1, char2)
        b[bigram] = b.get(bigram, 0) + 1
sorted_bigrams = sorted(b.items(), key=lambda x: x[1], reverse=True)


N = torch.zeros((27, 27), dtype=torch.int32)

all_chars =  ['.'] + sorted(list(set(''.join(names))))  
str_to_index = {ch: i for i, ch in enumerate(all_chars)}

index_to_str = {i: ch for i, ch in enumerate(all_chars)}

for name in names: 
    chs = ['.'] + list(name) + ['.'] # We do this to add <S> to first token and <E> to last token as to make a pair in starting and ending of the name
    for char1, char2 in zip(chs, chs[1:]):
        i1 = str_to_index[char1]
        i2 = str_to_index[char2]
        N[i1, i2] += 1


P = N.float() / N.float().sum(1, keepdim=True) # Normalize the rows to get probabilities


for i in range(10): # This will print 10 samples
    out = []
    ix = 0 # Start with the first character
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, replacement=True).item() # Sample the next character index
        out.append(index_to_str[ix])
        if ix == 0: # If we hit the end character, break the loop
            break
    print(''.join(out))


log_likelihood = 0.0
n = 0
for name in names[:3]: 
    chs = ['.'] + list(name) + ['.'] # We do this to add <S> to first token and <E> to last token as to make a pair in starting and ending of the name
    for char1, char2 in zip(chs, chs[1:]):
        i1 = str_to_index[char1]
        i2 = str_to_index[char2]
        prob = P[i1, i2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        print(f'{char1}{char2}: {prob:.4f} {logprob:.4f}') # Print the probability and log probability of each bigram
print(f'Log likelihood: {log_likelihood:.4f}') 
null = -log_likelihood
print(f'Null: {null/n}') # Print the null value


# creating a training set of bigrams

