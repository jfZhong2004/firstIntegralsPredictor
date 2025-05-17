import os
import torch
import argparse
from train import TransformerModel, MathExpressionDataset

class ModelParams:
    def __init__(self, emb_dim, n_heads, num_layers, dropout):
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.dropout = dropout

def load_model(model_path, vocab, params, device):
    model = TransformerModel(
        params=params,
        id2word={v: k for k, v in vocab.items()},
        is_encoder=True,
        with_output=True
    ).to(device)
    
    # Explicitly set weights_only=True to avoid security risks
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_expression(expression, vocab, max_len):
    tokens = ['<sos>'] + expression.split() + ['<eos>']
    input_ids = [vocab.get(t, vocab['<unk>']) for t in tokens]
    input_ids = input_ids[:max_len] + [vocab['<pad>']] * (max_len - len(input_ids))
    return torch.LongTensor(input_ids).unsqueeze(0)  # Add batch dimension

def decode_output(output_ids, id2word):
    tokens = [id2word[idx] for idx in output_ids if idx not in {0, 1, 2}]  # Exclude <pad>, <sos>, <eos>
    return ' '.join(tokens)

def main():
    parser = argparse.ArgumentParser(description='Test Math Expression Transformer Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--data_path', type=str, default='dataset.txt', help='Path to the dataset file (for vocab)')
    parser.add_argument('--emb_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load dataset to get vocab
    dataset = MathExpressionDataset(args.data_path, max_length=args.max_len)
    vocab = dataset.vocab
    id2word = {v: k for k, v in vocab.items()}

    # Create model parameters
    params = ModelParams(
        emb_dim=args.emb_dim,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    # Load model
    model = load_model(args.model_path, vocab, params, device)

    print("Model loaded. Enter a mathematical expression to test (type 'exit' to quit):")
    while True:
        user_input = input("Input: ").strip()
        if user_input.lower() == 'exit':
            break

        # Preprocess input
        input_tensor = preprocess_expression(user_input, vocab, args.max_len).to(device)

        # Generate output
        with torch.no_grad():
            output = model(input_tensor, input_tensor[:, :-1])  # Teacher forcing
            output_ids = torch.argmax(output, dim=-1).squeeze(0).tolist()

        # Decode output
        generated_expression = decode_output(output_ids, id2word)
        print(f"Output: {generated_expression}")

if __name__ == '__main__':
    main()