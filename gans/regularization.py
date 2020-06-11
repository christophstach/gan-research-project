import torch

def orthogonal_regularization(model, device):
    with torch.enable_grad():
        reg = 1e-6
        orth_loss = torch.zeros(1, device=device)
        
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0], device=device)
                orth_loss = orth_loss + (reg * sym.abs().sum())

    return orth_loss