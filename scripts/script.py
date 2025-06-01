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

g = torch.Generator().manual_seed(2147483647) # This is the seed for the random number generator

for i in range(10): # This will print 10 samples
    out = []
    ix = 0 # Start with the first character
    while True:
        ix = torch.multinomial(P[ix], num_samples=1, replacement=True, generator=g).item() # Sample the next character index
        out.append(index_to_str[ix])
        if ix == 0: # If we hit the end character, break the loop
            break
    print(''.join(out))