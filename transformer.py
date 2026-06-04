import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, f"d_model {d_model} not divisible by num_heads {num_heads}"
        self.d_k = d_model // num_heads
    
        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)
        
        self.out_linear = nn.Linear(d_model,d_model)
    
    def forward(self,k,v,q,mask=None):
    # 'q', 'k', 'v' all have shape: (batch_size, seq_len, d_model)
            
        # Linear projection pass
        query = self.q_linear(q)
        key = self.k_linear(k)
        value = self.v_linear(v)
        
        # Convert to (batch size, num_heads, seq_len, d_k)
        batch_size, tgt_len, _ = query.size()
        _, src_len, _          = key.size()
        query = query.view(batch_size, tgt_len, self.num_heads, self.d_k).permute(0,2,1,3)
        key   = key.view(batch_size,   src_len, self.num_heads, self.d_k).permute(0,2,1,3)
        value = value.view(batch_size, src_len, self.num_heads, self.d_k).permute(0,2,1,3)
        
        # Raw attention score
        scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(self.d_k) # Scores: (batch size, num_heads,seq_len,seq_len)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
            
        # Weights
        attention_weights = torch.softmax(scores,dim=-1)
        
        # multiply by V
        attention_output = torch.matmul(attention_weights,value) #(batch size, num_heads, seq_len, d_k)
        
        # Back to Original shape
        attention_output = attention_output.permute(0,2,1,3).contiguous()
        attention_output = attention_output.view(batch_size,tgt_len,self.d_model)
        
        return self.out_linear(attention_output)
        
class TransformerBlock(nn.Module):
    # Pre-LN variant: LayerNorm before each sublayer, not after.
    # Original "Attention is All You Need" uses Post-LN (norm after residual).
    # Pre-LN is more numerically stable; a final LayerNorm after the stack compensates.
    def __init__(self,num_heads,d_model,d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model=d_model,num_heads=num_heads) 
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
            )
        
    def forward(self,x,mask=None):
        norm_x = self.norm1(x)
        out1 = x + self.dropout1(self.attention(k=norm_x,v=norm_x,q=norm_x,mask=mask))
        output = out1 + self.dropout2(self.ffn(self.norm2(out1)))
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        
        pe = torch.zeros(max_len,d_model)
        
        position = torch.arange(0,max_len,dtype = torch.float).unsqueeze(1) #(max_len,1)
        
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x = x+self.pe[:,:x.size(1)]
        return x
    
class TransformerInput(nn.Module):
    def __init__(self,vocab_size,d_model,max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encode = PositionalEncoding(d_model=d_model,max_len=max_len)
        self.dropout = nn.Dropout(p=0.1)
    def forward(self,x):
        x = self.embedding(x)
        x = x*math.sqrt(self.d_model)
        x = self.pos_encode(x)
        x= self.dropout(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,transformer_num,num_heads,d_model,d_ff,vocab_size,max_len=5000):
        super().__init__()
        self.input_layer = TransformerInput(vocab_size=vocab_size,d_model=d_model,max_len=max_len)
        self.transformers = nn.ModuleList([TransformerBlock(num_heads=num_heads,d_model=d_model,d_ff=d_ff) for i in range(transformer_num)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self,x,mask=None):
        x = self.input_layer(x)

        for transformer in self.transformers:
            x = transformer(x,mask=mask)
        return self.norm(x)
    
class TransformerDecoderBlock(nn.Module):
    # Pre-LN variant — see TransformerBlock comment.
    def __init__(self,num_heads,d_model,d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.attention1 = MultiHeadAttention(d_model=d_model,num_heads=num_heads) 
        self.attention2 = MultiHeadAttention(d_model=d_model,num_heads=num_heads) 
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
        )
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
    def forward(self,x,encoder_output,tgt_mask=None,src_mask=None):
        norm_x = self.norm1(x)
        out1  = x    + self.dropout1(self.attention1(k=norm_x,v=norm_x,q=norm_x,mask=tgt_mask))
        cross = out1 + self.dropout2(self.attention2(k=encoder_output,v=encoder_output,q=self.norm2(out1),mask=src_mask))
        output = cross + self.dropout3(self.ffn(self.norm3(cross)))
        return output
    
class TransformerDecoder(nn.Module):
    def __init__(self,transformer_num,num_heads,d_model,d_ff,vocab_size,max_len=5000):
        super().__init__()
        self.input_layer = TransformerInput(vocab_size=vocab_size,d_model=d_model,max_len=max_len)
        self.transformers = nn.ModuleList([TransformerDecoderBlock(num_heads = num_heads,d_model=d_model,d_ff=d_ff) for i in range(transformer_num)])
        self.norm = nn.LayerNorm(d_model)
        self.linear_out = nn.Linear(d_model,vocab_size)

    def forward(self,x,encoder_output,tgt_mask=None,src_mask=None):
        x = self.input_layer(x)
        for transformer in self.transformers:
            x = transformer(x=x,encoder_output=encoder_output,src_mask=src_mask,tgt_mask=tgt_mask)

        out = self.linear_out(self.norm(x))
        return out

    
class Transformer(nn.Module):
    def __init__(self,encoder_num,decoder_num,num_heads,d_model,d_ff,vocab_size,max_len):
        super().__init__()
        self.encoder = TransformerEncoder(transformer_num=encoder_num,num_heads=num_heads,d_ff=d_ff,d_model=d_model,vocab_size=vocab_size,max_len=max_len)
        self.decoder = TransformerDecoder(transformer_num=decoder_num,num_heads=num_heads,d_model=d_model,d_ff=d_ff,vocab_size=vocab_size,max_len=max_len)
        
    def forward(self,src,tgt,tgt_mask=None,src_mask = None):  
        enc_output = self.encoder(src,mask=src_mask)    
        output = self.decoder(tgt,encoder_output=enc_output,tgt_mask=tgt_mask,src_mask=src_mask)
        return output