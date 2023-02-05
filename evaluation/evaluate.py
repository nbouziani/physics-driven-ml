import torch
import firedrake as fd

from tqdm.auto import tqdm, trange


fd_backend = fd.get_backend()

def evaluate(model, config, data, V):

    if len(data) != 1:
        raise NotImplementedError("Evaluate on more than 1 sample necessitates defining an appropriate evaluation metric.")

    (k_exact, _, u_obs), = data

    model.eval()

    u_obs = fd_backend.to_ml_backend(u_obs)

    with torch.no_grad():
        kP = model(u_obs)
        kF = fd_backend.from_ml_backend(kP, V)
        error = fd.norm(kF - k_exact, norm_type=config.evaluation_metric)

    return error
