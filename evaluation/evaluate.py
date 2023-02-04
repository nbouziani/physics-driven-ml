import torch
import firedrake as fd

from tqdm.auto import tqdm, trange


fd_backend = fd.get_backend()

def evaluate(model, V, config, data, metric="L2"):

    model.eval()

    y_exact, = data
    y_exact = fd.to_ml_backend(y_exact)

    with torch.no_grad():
        yP = model(data)
        yF = fd.from_ml_backend(yP, V) 
        fd.norm(yF - y_exact, norm_type=metric)
