import torch
import math
from captum.attr import Occlusion, Saliency

def get_explanation_pdt(image: torch.Tensor, model: torch.nn.Module, label: int = 0, exp: str = "saliency"):
    image = image.requires_grad_(True)
    model.zero_grad()

    def agg_segmentation_wrapper(inp):
        model_out = model(inp)  
        out_max = model_out.argmax(dim=1, keepdim=True)
        selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)
        aggregated_logits = (model_out * selected_inds).sum(dim=(2, 3, 4))  

        return aggregated_logits
    
    # ----------- parameters -----------
    size = 28
    sliding_window_shapes = (1, size, size, 2) 
    strides = (1, size, size, 1)               
    baselines = image.mean().item() # 0                          
    perturbations_per_eval = 1

    out = agg_segmentation_wrapper(image)
    pdtr = out[:, label]
    pdt = torch.sum(out[:, label])

    if exp.lower() == "saliency":
        saliency_function = Saliency(agg_segmentation_wrapper)
        expl = saliency_function.attribute(image, target=1, abs=True)
    elif exp.lower() == "occlusion":
        occlusion_function = Occlusion(agg_segmentation_wrapper)
        expl = occlusion_function.attribute(
            image,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            target=label,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=True
        )
    else:
        raise NotImplementedError("Explanation not implemented for: ", exp)
    
    return expl, pdtr

def sample_eps_Inf_3d(image: torch.Tensor, epsilon: float, N: int) -> torch.Tensor:
    """Generate random perturbations for 3D images."""
    C, D, H, W = image.shape[-4:]  # Get last 4 dimensions
    
    return torch.empty(N, C, D, H, W, 
                      device=image.device, 
                      dtype=image.dtype).uniform_(-epsilon, epsilon)

def get_explanation_sensitivity():
    max_diff = -math.inf
    for _ in range(sen_N):
        sample = torch.FloatTensor(sample_eps_inf(X, sen_r, 1)).cuda()
    pass

def evaluate_metrics(
        model: torch.nn.Module, 
        data_loader: torch.utils.data.DataLoader, 
        pert: str = "Gaussian",
        exp: str = "Saliency"
    ):
    if pert == "Gauusian":
        binary_I = True
    else:
        raise NotImplementedError("This perturbation is not implemented.")
    
    model.eval()
    infidelities = []
    max_sensitivities = []

    for i, batch in enumerate(data_loader):
        if i >= 5: # TODO: Set higher after debug
            break

        X = batch["iamge"].cuda()
        label = 1 #batch["label"].cuda().float()

        expl, pdt = get_explanation_pdt(X, model, label, exp)

        norm = torch.linalg.norm(expl)

        infid = get_exp_infidelity()
        infidelities.append(infid)

        sensitivity = get_explanation_sensitivity()

