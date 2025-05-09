import torch
import torch.nn as nn
import math

class LayerNorm(nn.Module):
    def __init__(self, epsilon: float = 10**-6):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.bias = nn.Parameter(torch.zeros(1)) 
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std_dev = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean) / (std_dev + self.epsilon) + self.bias

class MultiHeadAttention(nn.Module):
    '''
    Section 3.2.3
    '''

class FeedForward(nn.Module):
    '''
    Section 3.3 
    Processes each word in the sentence independently
    FFN(x) = max(0,x W1 + b1) W2 + b2
    Input --> (batch, seq_len, d_model) 
    Apply linear layer 1: 
    d_model --> d_ff
    ReLU activation is applied element-wise after this
    Apply linear layer 2:
    d_ff --> d_model
    
    '''

    def __init__(self, d_model:int, d_ff:int, dropout: float):
        super().__init__()
        self.linear_layer1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer2 = nn.Linear(d_ff, d_model)

    def forward(self,x):
        output_layer1 = self.linear_layer1(x)
        relu = torch.relu(output_layer1)
        dropout_layer1 = self.dropout(relu)
        output_layer2 = self.linear_layer2(dropout_layer1)
        return output_layer2

class input_embedding(nn.Module):
    '''
    Section 3.4
    d_model: dimension of the embedding vector
    vocab_size: size of the vocabulary (number of words)
    Output: Input embedding vector of size d_model
    '''
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)  
    
class postional_encoding(nn.Module):
    '''
    Section 3.5
    Some information about the relative or absolute position of the tokens in the sequence.

    seq_len: maximum length of the sequence
    Output: vector of size as input embeddings
    
    '''
    def __init__(self, d_model:int, seq_len:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.droput = nn.Dropout(dropout)

        ''' 
        each word in the sequence will have a encoding of size d_model. (seq_len x d_model)
        calculating denominator using fact that: a^b = exp(b * ln(a))
        pos[i] = index of token in sequence (0 to seq_len - 1)
        denominator[i] = 1 / (10000 ** (2i / d_model)) using exp(-log(10000) * i / d_model)
        '''
        pe = torch.zeros(seq_len, d_model)
        pos = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,d_model,2).float()* (-math.log(10000.0)/d_model))
        pe[0][0::2] = torch.sin(pos*denominator)
        pe[0][1::2] = torch.cos(pos * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe) # Save the tensor as a file; not trainable

    def forward(self, x):
        '''
        Add positional encoding to each word in the sentence.
        '''
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.droput(x)

