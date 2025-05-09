# Text Translation Using Transformers

Making transformer based model for English to Hindi translation task from scratch using PyTorch based on paper -> "Attention is all you need"

## Variable values:
    d_model = 512
    d_ff = 2048
    h = 8
## Input Embedding:
converts all words in the input sentence to its d_model dimensional embedding vector

## Positional Encoding:
These encodings stores information about the relative or absolute position of the tokens in the sequence.
* **Encoding Formula**:

  * For even indices (2i):
    $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$
  * For odd indices (2i+1):
    $PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$

Both these embeddings are added element wise. 

## Layer Norm:
Formula: 
  $\text{LayerNorm}(x) = \alpha \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
## FeedForward Network
 This consists of two linear transformations with a ReLU activation in between.
    Processes each word in the sentence independently
 $ \text{FFN}(x) = \text{Linear}_2(\text{ReLU}(\text{Linear}_1(x)))
    $

    Input: (batch_size, seq_len, d_model)
    Layer 1: d_model → d_ff
    ReLU
    Layer 2: d_ff → d_model
    Dropout is applied between the layers for regularization.

