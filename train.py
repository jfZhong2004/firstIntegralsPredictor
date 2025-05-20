# train.py
import os
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, params, id2word, is_encoder=True, with_output=True):
        super().__init__()
        self.vocab_size = len(id2word)
        self.emb_dim = params.emb_dim
        self.n_heads = params.n_heads
        self.num_layers = params.num_layers
        self.dropout = params.dropout
        self.pad_token_id = 0  # Assuming <pad> is 0

        # Embedding layers
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=self.pad_token_id)
        self.positional_encoding = PositionalEncoding(self.emb_dim, dropout=params.dropout)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=self.emb_dim,
            nhead=self.n_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dropout=params.dropout,
            batch_first=True
        )

        # Final linear layer
        self.fc_out = nn.Linear(self.emb_dim, self.vocab_size)

    def forward(self, src, tgt):
        # src: (batch, src_seq_len), tgt: (batch, tgt_seq_len)
        src_emb = self.positional_encoding(self.embedding(src))
        tgt_emb = self.positional_encoding(self.embedding(tgt))
        
        # Generate masks
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_padding_mask = (src == self.pad_token_id)
        tgt_padding_mask = (tgt == self.pad_token_id)
        
        # Transformer forward
        output = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.fc_out(output)

class MathExpressionDataset(Dataset):
    def __init__(self, file_path, max_length=512):
        self.data = []
        self.vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        self.max_length = max_length
        
        # Build vocabulary
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                io_pair = line.strip().split(';')
                if len(io_pair) != 2:
                    continue
                
                for expr in io_pair[0].split(',') + io_pair[1].split(','):
                    tokens = expr.strip().split()
                    for token in tokens:
                        if token not in self.vocab:
                            self.vocab[token] = len(self.vocab)
        
        # Load data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                inputs, outputs = line.strip().split(';')
                input_exprs = [s.strip() for s in inputs.split(',')]
                output_exprs = [s.strip() for s in outputs.split(',')]
                
                input_ids = []
                for expr in input_exprs:
                    tokens = ['<sos>'] + expr.split() + ['<eos>']
                    ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
                    input_ids.extend(ids)
                
                output_ids = []
                for expr in output_exprs:
                    tokens = ['<sos>'] + expr.split() + ['<eos>']
                    ids = [self.vocab.get(t, self.vocab['<unk>']) for t in tokens]
                    output_ids.extend(ids)
                
                # Truncate/pad sequences
                input_ids = input_ids[:self.max_length] + [self.vocab['<pad>']] * (self.max_length - len(input_ids))
                output_ids = output_ids[:self.max_length] + [self.vocab['<pad>']] * (self.max_length - len(output_ids))
                
                self.data.append({
                    'input_ids': torch.LongTensor(input_ids),
                    'output_ids': torch.LongTensor(output_ids)
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")  # 添加此行以打印设备信息
    
    dataset = MathExpressionDataset(args.data_path, max_length=args.max_len)
    model = TransformerModel(
        params=args,
        id2word={v:k for k,v in dataset.vocab.items()},
        is_encoder=True,
        with_output=True
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss(ignore_index=dataset.vocab['<pad>'])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            targets = batch['output_ids'].to(device)
            
            optimizer.zero_grad()
            
            # Shift target for teacher forcing
            outputs = model(inputs, targets[:, :-1])
            loss = criterion(outputs.reshape(-1, len(dataset.vocab)), targets[:, 1:].reshape(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{args.num_epochs}], Loss: {avg_loss:.4f}')
        
        if (epoch+1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Math Expression Transformer Training')
    parser.add_argument('--data_path', type=str, default='dataset.txt')
    parser.add_argument('--save_dir', type=str, default='saved_models')
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)