import os
from argparse import ArgumentParser
import torch
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader

from pl_tsp_model import TSPModel
from pl_mis_model import MISModel

def arg_parser():
    parser = ArgumentParser(description='Train a PyTorch diffusion model on a TSP dataset.')
    parser.add_argument('--task', type=str, default='tsp')
    parser.add_argument('--storage_path', type=str, default='')
    parser.add_argument('--training_split', type=str, default='data/tsp/tsp50_test_concorde1.txt')
    parser.add_argument('--validation_split', type=str, default='data/tsp/tsp50_test_concorde1.txt')
    parser.add_argument('--test_split', type=str, default='data/tsp/tsp50_test_concorde1.txt')
    parser.add_argument('--validation_examples', type=int, default=8)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lr_scheduler', type=str, default="cosine-decay")

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--use_activation_checkpoint', action='store_true')

    parser.add_argument('--diffusion_type', type=str, default='categorical')
    parser.add_argument('--diffusion_schedule', type=str, default='linear')
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--inference_diffusion_steps', type=int, default=50)
    parser.add_argument('--inference_schedule', type=str, default='cosine')
    parser.add_argument('--inference_trick', type=str, default="ddim")
    parser.add_argument('--sequential_sampling', type=int, default=1)
    parser.add_argument('--parallel_sampling', type=int, default=1)

    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--sparse_factor', type=int, default=-1)
    parser.add_argument('--aggregation', type=str, default='sum')
    parser.add_argument('--two_opt_iterations', type=int, default=1000)
    parser.add_argument('--save_numpy_heatmap', action='store_true')

    parser.add_argument('--project_name', type=str, default='tsp_diffusion')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_logger_name', type=str, default="tsp_diffusion_graph_categorical_tsp50")
    parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
    parser.add_argument('--ckpt_path', type=str, default='ckpt/tsp50_categorical.ckpt')
    parser.add_argument('--resume_weight_only', action='store_true')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', default=True)
    parser.add_argument('--do_valid_only', action='store_true')
    parser.add_argument('--use_ddp', default=True)
    parser.add_argument('--constraint_type', default='basic')
    parser.add_argument('--f_name', default='test_for_jk')

    args = parser.parse_args()
    return args

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(model, train_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = [b.to(device) for b in batch]
            loss = model.training_step(batch, batch_idx=0)  # batch_idx can be ignored
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

def evaluate_model(model, data_loader, device, split="test"):
    model.eval()
    metrics = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = [b.to(device) for b in batch]
            batch_metrics = model.test_step(batch, batch_idx, split=split)
            for key, value in batch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

    # Average the metrics
    averaged_metrics = {key: torch.tensor(values, dtype=torch.float).mean().item() for key, values in metrics.items()}

    print(f"{split.capitalize()} Metrics: {averaged_metrics}")
    return averaged_metrics

def main(args):
    seed = 2025
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == 'tsp':
        model_class = TSPModel
    elif args.task == 'mis':
        model_class = MISModel
    else:
        raise NotImplementedError

    model = model_class(param_args=args).to(device)

    if args.ckpt_path and os.path.exists(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    if args.do_train:
        train_dataset = model.train_dataset
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )

        train_model(model, train_loader, optimizer, device, args.num_epochs)

    if args.do_test:
        test_dataset = model.test_dataset
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
        )
        test_metrics = evaluate_model(model, test_loader, device, split="test")

    result = pd.DataFrame(model.tsp_solutions, columns=['sample_idx', 'solved_cost', 'tour'])
    f_name = f'results/{args.f_name}.csv'
    result.to_csv(f_name, index=0)
    print(f"Results saved to {f_name}")

if __name__ == '__main__':
    args = arg_parser()
    main(args)
