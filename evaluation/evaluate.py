import torch
import firedrake as fd

from tqdm.auto import tqdm


fd_backend = fd.get_backend()

def evaluate(model, config, data, V):

    model.eval()

    eval_steps = min(len(data), config.max_eval_steps)
    total_error = 0.0
    for step_num, batch in tqdm(enumerate(data[:eval_steps]), total=eval_steps):

        # TODO: Add device to batch
        # Convert to PyTorch tensors
        k_exact, _, u_obs = batch
        u_obs = fd_backend.to_ml_backend(u_obs)

        with torch.no_grad():
            kP = model(u_obs)
            kF = fd_backend.from_ml_backend(kP, V)
            total_error += fd.norm(kF - k_exact, norm_type=config.evaluation_metric)

    total_error /= eval_steps
    return total_error
