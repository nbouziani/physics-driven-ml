import os
import argparse

import torch
import firedrake as fd
import firedrake.ml as fd_ml

from torch.utils.data import DataLoader

from functools import partial
from tqdm.auto import tqdm

from physics_driven_ml.models import EncoderDecoder, CNN
from physics_driven_ml.utils import ModelConfig, get_logger
from physics_driven_ml.dataset_processing import PDEDataset, BatchedElement


fd_backend = fd_ml.load_backend()


def evaluate(model, config, dataloader, disable_tqdm=False):
    """Evaluate the model on a given dataset."""

    model.eval()

    eval_steps = min(len(dataloader), config.max_eval_steps)
    total_error = 0.0
    compute_error = partial(eval_error, evaluation_metric=config.evaluation_metric)
    for step_num, batch in tqdm(enumerate(dataloader), total=eval_steps, disable=disable_tqdm):

        # Move batch to device
        batch = BatchedElement(*[x.to(config.device, non_blocking=True) if isinstance(x, torch.Tensor) else x for x in batch])
        u_obs = batch.u_obs
        k_exact, = batch.target_fd

        with torch.no_grad():
            kP = model(u_obs)
            kF = fd_backend.from_ml_backend(kP, k_exact.function_space())
            total_error += compute_error(kF, k_exact)

        if step_num == eval_steps - 1:
            break

    total_error /= eval_steps
    return total_error


def eval_error(x, x_exact, evaluation_metric):
    """Compute the error between x and x_exact for a given metric."""
    if evaluation_metric == "avg_rel":
        # Compute relative L2-error: ||x - x_exact||_{L2}^{2} / ||x_exact||_{L2}^{2}
        return fd.assemble((x - x_exact) ** 2 * fd.dx)/fd.assemble(x_exact ** 2 * fd.dx)
    return fd.norm(x - x_exact, norm_type=evaluation_metric)


if __name__ == "__main__":
    logger = get_logger("Evaluation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ["DATA_DIR"], type=str, help="Data directory")
    parser.add_argument("--model", default="cnn", type=str, help="one of [encoder-decoder, cnn]")
    parser.add_argument("--model_dir", default="model", type=str, help="Directory name to load the model from")
    parser.add_argument("--model_version", default="", type=str, help="Saved model version to load (e.g. for a specific checkpoint)")
    parser.add_argument("--max_eval_steps", default=5000, type=int, help="Maximum number of evaluation steps")
    parser.add_argument("--evaluation_metric", default="L2", type=str, help="Evaluation metric: one of [Lp, H1, Hdiv, Hcurl, avg_rel]")
    parser.add_argument("--dataset", default="heat_conductivity", type=str, help="Dataset name")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--device", default="cpu", type=str, help="Device identifier (e.g. 'cuda:0' or 'cpu')")
    parser.add_argument("--eval_set", default="test", type=str, help="Dataset split to evaluate on")

    args = parser.parse_args()
    config = ModelConfig(**dict(args._get_kwargs()))

    # Load dataset
    dataset_dir = os.path.join(args.data_dir, "datasets", args.dataset)
    logger.info(f"Loading dataset from {dataset_dir}\n")
    dataset = PDEDataset(dataset=config.dataset, dataset_split=args.eval_set, data_dir=config.data_dir)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=dataset.collate, shuffle=False)

    # Load model
    model_dir = os.path.join(args.data_dir, "saved_models", args.model_dir, args.model_version)

    logger.info(f"Loading model checkpoint from {model_dir}\n")
    if args.model == "encoder-decoder":
        model = EncoderDecoder.from_pretrained(model_dir)
    elif args.model == "cnn":
        model = CNN.from_pretrained(model_dir)
    # Set double precision (default Firedrake type)
    model.double()
    # Move model to device
    model.to(config.device)

    # Evaluate model
    error = evaluate(model, config, dataloader)
    logger.info(f"\n\t Error (metric: {config.evaluation_metric}): {error:.4e}")
