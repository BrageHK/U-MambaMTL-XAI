import torch
from torchvision import transforms
from monai.transforms.intensity.array import ScaleIntensityRangePercentiles
import numpy as np
from tqdm import tqdm
from pytorch_grad_cam import AblationCAM
from captum.metrics import infidelity, sensitivity_max
from captum.attr import Saliency

def channel_wise_normalize_and_clamp(x):
    for i in range(3):
        #x[i] = torch.clamp(x[i] + 0.5 - x[i].mean(), min=-0.999, max=0.999)
        x[i] = ScaleIntensityRangePercentiles(lower=.1, upper=99.9, b_min=-1, b_max=1, clip=True)(x[i])
    return x

def normalize_and_clamp(x):
    print("Normalizing")
    print(x.min())
    print(x.max())
    #x = torch.clamp(x + 0.5 - x.mean(), min=-0.999, max=0.999)
    x = ScaleIntensityRangePercentiles(lower=.1, upper=99.9, b_min=-1, b_max=1, clip=True)(x)
    print("Done normalizing")
    return x

class SemanticSegmentationTarget3D:
    """Target for 3D semantic segmentation - computes weighted sum over a mask region."""
    def __init__(self, category, mask, device="cuda"):
        self.category = category
        self.mask = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
        if device == "cuda" and torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # model_output shape: (B, C, H, W, D) or (C, H, W, D)
        if model_output.dim() == 5:
            return (model_output[:, self.category, :, :, :] * self.mask).sum()
        else:
            return (model_output[self.category, :, :, :] * self.mask).sum()


class ModelWrapper3D(torch.nn.Module):
    """Wrapper to handle 3D->2D->3D conversion for grad-cam compatibility."""
    def __init__(self, model, slice_dim=-1):
        super().__init__()
        self.model = model
        self.slice_dim = slice_dim
        self._current_slice = None
        self._full_input_shape = None

    def set_slice(self, slice_idx, full_shape):
        self._current_slice = slice_idx
        self._full_input_shape = full_shape

    def forward(self, x):
        # x is 2D: (B, C, H, W) - expand to 3D for model
        if self._current_slice is not None and x.dim() == 4:
            # Create a zero tensor with full 3D shape
            B, C, H, W = x.shape
            D = self._full_input_shape[-1]
            x_3d = torch.zeros(B, C, H, W, D, device=x.device, dtype=x.dtype)
            x_3d[:, :, :, :, self._current_slice] = x
            out = self.model(x_3d)
            # Return only the slice we care about
            return out[:, :, :, :, self._current_slice]
        return self.model(x)


def compute_ablation_cam_3d(
    model,
    input_tensor,
    target_category,
    target_mask=None,
    target_layers=None,
    device="cuda",
    batch_size=16,
    ratio_channels_to_ablate=0.5
):
    """
    Compute AblationCAM for 3D medical segmentation models.

    Since pytorch-grad-cam is 2D-oriented, this function processes
    the volume slice-by-slice and aggregates the results.

    Args:
        model: The segmentation model (LitModel)
        input_tensor: Input tensor of shape (B, C, H, W, D)
        target_category: Class index to explain (e.g., 1 for PCa)
        target_mask: Optional mask to focus attribution (H, W, D) or None for full volume
        target_layers: List of layers to use for CAM. If None, uses encoder last stage
        device: Device to run on
        batch_size: Batch size for ablation perturbations
        ratio_channels_to_ablate: Fraction of channels to ablate per forward pass

    Returns:
        cam_volume: Attribution map of shape (H, W, D)
    """
    model.eval()

    # Get the underlying model (handle LitModel wrapper)
    if hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model

    # Default target layers: last encoder stage and mamba layer
    if target_layers is None:
        target_layers = [base_model.encoder.stages[-1]]

    B, C, H, W, D = input_tensor.shape

    # Create mask if not provided (use prediction as mask)
    if target_mask is None:
        with torch.no_grad():
            output = model(input_tensor)
            pred = (torch.sigmoid(output[:, target_category]) > 0.5).float()
            target_mask = pred[0].cpu().numpy()

    # Process slice by slice for compatibility
    cam_slices = []

    for slice_idx in tqdm(range(D), desc="Computing AblationCAM per slice"):
        # Extract 2D slice
        slice_2d = input_tensor[:, :, :, :, slice_idx]  # (B, C, H, W)
        mask_2d = target_mask[:, :, slice_idx] if target_mask.ndim == 3 else target_mask

        # Skip slices with no mask content
        if isinstance(mask_2d, np.ndarray):
            mask_sum = mask_2d.sum()
        else:
            mask_sum = mask_2d.sum().item()

        if mask_sum < 1:
            cam_slices.append(np.zeros((H, W), dtype=np.float32))
            continue

        # Create wrapper for this slice
        wrapper = ModelWrapper3D(model, slice_dim=-1)
        wrapper.set_slice(slice_idx, input_tensor.shape)
        wrapper.to(device)

        # Get target layers from wrapper
        if hasattr(wrapper.model, 'model'):
            wrapper_target_layers = [wrapper.model.model.encoder.stages[-1]]
        else:
            wrapper_target_layers = [wrapper.model.encoder.stages[-1]]

        # Create 2D target
        class Target2D(torch.nn.Module):
            def __init__(self, category, mask_2d):
                super().__init__()
                self.category = category
                mask_tensor = torch.from_numpy(mask_2d).float() if isinstance(mask_2d, np.ndarray) else mask_2d.float()
                self.register_buffer('mask', mask_tensor.to(device))

            def forward(self, output):
                # output: (B, C, H, W)
                return (output[:, self.category] * self.mask).sum()

        target = Target2D(target_category, mask_2d)

        try:
            with AblationCAM(
                model=wrapper,
                target_layers=wrapper_target_layers,
                batch_size=batch_size,
                ratio_channels_to_ablate=ratio_channels_to_ablate
            ) as cam:
                grayscale_cam = cam(
                    input_tensor=slice_2d,
                    targets=[target]
                )[0]  # (H, W)
            cam_slices.append(grayscale_cam)
        except Exception as e:
            print(f"Warning: AblationCAM failed for slice {slice_idx}: {e}")
            cam_slices.append(np.zeros((H, W), dtype=np.float32))

    # Stack slices into volume
    print("made ", len(cam_slices), " slices")
    cam_volume = np.stack(cam_slices, axis=-1)  # (H, W, D)

    return cam_volume


def compute_ablation_cam_3d_direct(
    model,
    input_tensor,
    target_category,
    target_mask=None,
    ablation_size=(16, 16, 4),
    stride=None,
    device="cuda"
):
    """
    Compute AblationCAM directly on 3D volume using spatial ablation.

    This is a direct 3D implementation that ablates spatial regions
    rather than channels, similar to occlusion sensitivity but using
    the ablation principle.

    Args:
        model: The segmentation model
        input_tensor: Input tensor of shape (B, C, H, W, D)
        target_category: Class index to explain
        target_mask: Optional mask to focus attribution
        ablation_size: Size of ablation window (h, w, d)
        stride: Stride for ablation window. If None, uses ablation_size
        device: Device to run on

    Returns:
        cam_volume: Attribution map of shape (H, W, D)
    """
    model.eval()

    if stride is None:
        stride = ablation_size

    B, C, H, W, D = input_tensor.shape
    ah, aw, ad = ablation_size
    sh, sw, sd = stride if isinstance(stride, tuple) else (stride, stride, stride)

    # Get baseline output
    with torch.no_grad():
        baseline_output = model(input_tensor)
        if target_mask is not None:
            mask_tensor = torch.from_numpy(target_mask).float().to(device) if isinstance(target_mask, np.ndarray) else target_mask.float().to(device)
            baseline_score = (torch.sigmoid(baseline_output[:, target_category]) * mask_tensor).sum().item()
        else:
            baseline_score = torch.sigmoid(baseline_output[:, target_category]).sum().item()

    # Initialize attribution map
    attribution = np.zeros((H, W, D), dtype=np.float32)
    counts = np.zeros((H, W, D), dtype=np.float32)

    # Compute number of windows
    n_h = (H - ah) // sh + 1
    n_w = (W - aw) // sw + 1
    n_d = (D - ad) // sd + 1

    total_windows = n_h * n_w * n_d

    with torch.no_grad():
        for i, h_start in enumerate(tqdm(range(0, H - ah + 1, sh), desc="AblationCAM 3D")):
            for w_start in range(0, W - aw + 1, sw):
                for d_start in range(0, D - ad + 1, sd):
                    # Create ablated input (set region to zero or mean)
                    ablated_input = input_tensor.clone()
                    ablated_input[:, :, h_start:h_start+ah, w_start:w_start+aw, d_start:d_start+ad] = 0

                    # Get ablated output
                    ablated_output = model(ablated_input)
                    if target_mask is not None:
                        ablated_score = (torch.sigmoid(ablated_output[:, target_category]) * mask_tensor).sum().item()
                    else:
                        ablated_score = torch.sigmoid(ablated_output[:, target_category]).sum().item()

                    # Attribution = drop in score (positive means important)
                    importance = baseline_score - ablated_score

                    # Add to attribution map
                    attribution[h_start:h_start+ah, w_start:w_start+aw, d_start:d_start+ad] += importance
                    counts[h_start:h_start+ah, w_start:w_start+aw, d_start:d_start+ad] += 1

    # Average overlapping regions
    counts = np.maximum(counts, 1)
    attribution = attribution / counts

    return attribution


def perturb_fn(inputs):
    """
    Perturbation function for infidelity metric.
    Adds Gaussian noise to inputs.
    """
    noise = torch.randn_like(inputs) * 0.1
    return noise, inputs - noise


def create_agg_segmentation_wrapper(model):
    """
    Factory function to create aggregation wrapper for segmentation models.

    Aggregates spatial outputs by selecting the predicted class per pixel
    and summing the logits, producing output shape (B, C) suitable for
    Captum attribution methods.

    Args:
        model: Segmentation model that outputs (B, C, H, W, D) or (B, C, H, W)

    Returns:
        Wrapper function that returns aggregated logits (B, C)
    """
    def wrapper(inp):
        model_out = model(inp)
        out_max = model_out.argmax(dim=1, keepdim=True)
        selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)
        # Use mean over spatial dims to keep output in per-voxel logit range,
        # preventing infidelity from scaling with volume size (N^2).
        spatial_dims = tuple(range(2, model_out.dim()))
        aggregated_logits = (model_out * selected_inds).mean(dim=spatial_dims)
        return aggregated_logits
    return wrapper


def create_masked_agg_wrapper(model, target_class, mask=None, threshold=0.5):
    """
    Factory function to create masked aggregation wrapper for segmentation models.

    Aggregates outputs within a mask region (either provided or predicted).

    Args:
        model: Segmentation model
        target_class: Target class index for generating prediction mask
        mask: Optional pre-computed mask tensor. If None, uses predicted mask
        threshold: Sigmoid threshold for generating prediction mask

    Returns:
        Wrapper function that returns aggregated logits (B, C)
    """
    def wrapper(inp):
        model_out = model(inp)

        if mask is not None:
            # Use provided mask
            region_mask = mask.float()
            if region_mask.dim() < model_out.dim():
                # Expand mask to match output dimensions
                region_mask = region_mask.unsqueeze(0).unsqueeze(0)
        else:
            # Generate mask from predictions for target class
            region_mask = (torch.sigmoid(model_out[:, target_class:target_class+1]) > threshold).float()

        # Apply mask and aggregate with mean to keep output scale-invariant
        masked_out = model_out * region_mask
        spatial_dims = tuple(range(2, model_out.dim()))
        aggregated = masked_out.mean(dim=spatial_dims)
        return aggregated
    return wrapper


def compute_xai_metrics(model_fn, inputs, attributions, target, n_perturb_samples=10):
    """
    Compute Infidelity and Sensitivity metrics for attribution maps.

    Note: For segmentation models, use compute_xai_metrics_segmentation instead,
    which handles spatial output aggregation.

    Args:
        model_fn: The model forward function (should return scalar or (B, C) output)
        inputs: Input tensor
        attributions: Attribution map from explainability method
        target: Target class index
        n_perturb_samples: Number of perturbation samples for sensitivity

    Returns:
        dict with 'infidelity' and 'sensitivity' scores
    """
    # Compute Infidelity - measures how well attributions explain model predictions
    # Lower is better
    infid_score = infidelity(
        model_fn,
        perturb_fn,
        inputs,
        attributions,
        target=target,
        normalize=True
    )

    # Compute Sensitivity - measures attribution stability under input perturbations
    # Lower is better (more stable)
    sens_score = sensitivity_max(
        lambda x: Saliency(model_fn).attribute(x, target=target, abs=True),
        inputs,
        perturb_radius=0.02,
        n_perturb_samples=n_perturb_samples
    )

    return {
        'infidelity': infid_score.item() if infid_score.numel() == 1 else infid_score.mean().item(),
        'sensitivity': sens_score.item() if sens_score.numel() == 1 else sens_score.mean().item()
    }


def compute_sensitivity_manual(
    attr_fn,
    inputs,
    target,
    base_attributions=None,
    perturb_radius=0.02,
    n_perturb_samples=10,
):
    """
    Memory-efficient manual computation of sensitivity metric.

    Sensitivity = max over perturbations of ||attr(x) - attr(x+d)|| / ||attr(x)||

    Measures attribution stability under small input perturbations.
    Lower values indicate more stable/reliable attributions.

    Args:
        attr_fn: Attribution function that takes (inputs, target) and returns attributions
        inputs: Input tensor
        target: Target class index
        base_attributions: Pre-computed attributions for original input. If None, computed here
        perturb_radius: Maximum L-inf perturbation magnitude
        n_perturb_samples: Number of random perturbations to test

    Returns:
        Maximum sensitivity score (float)
    """
    inputs = inputs.detach().requires_grad_(True)

    # Compute base attributions if not provided
    if base_attributions is None:
        base_attributions = attr_fn(inputs, target)

    base_attributions = base_attributions.detach()

    # L2 norm of base attributions for normalization
    base_norm = torch.norm(base_attributions).item()
    if base_norm == 0:
        base_norm = 1.0

    max_sensitivity = 0.0

    for _ in range(n_perturb_samples):
        # Generate random perturbation within L-inf ball
        perturbation = torch.empty_like(inputs).uniform_(-perturb_radius, perturb_radius)
        perturbed_inputs = (inputs + perturbation).detach().requires_grad_(True)

        # Compute attributions for perturbed input
        perturbed_attributions = attr_fn(perturbed_inputs, target)

        # L2 norm of difference, normalized by base attribution norm
        diff = (perturbed_attributions - base_attributions).detach()
        sensitivity = torch.norm(diff).item() / base_norm

        max_sensitivity = max(max_sensitivity, sensitivity)

        # Clear memory
        del perturbed_inputs, perturbed_attributions, diff
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return max_sensitivity


def compute_infidelity_manual(
    model_fn,
    inputs,
    attributions,
    target,
    n_perturb_samples=10,
    perturb_scale=0.02,
    normalize=True
):
    """
    Memory-efficient manual computation of infidelity metric.

    Infidelity measures how well attributions explain model output changes.
    Lower values indicate attributions better explain the model.

    With normalize=True (default), performs optimal scaling:
        beta = E[df * (attr . delta)] / E[(attr . delta)^2]
        infidelity = E[(df - beta * attr . delta)^2]

    Without normalization:
        infidelity = E[(attr . delta - df)^2]

    Args:
        model_fn: Model forward function (should return (B, C) after aggregation)
        inputs: Input tensor
        attributions: Attribution map matching input shape
        target: Target class index
        n_perturb_samples: Number of perturbation samples
        perturb_scale: Scale of Gaussian perturbations
        normalize: If True, apply optimal scaling (beta) before scoring

    Returns:
        Infidelity score (float)
    """
    inputs = inputs.detach()
    attributions = attributions.detach()

    # Get baseline prediction
    with torch.no_grad():
        baseline_out = model_fn(inputs)
        if baseline_out.dim() > 1:
            baseline_score = baseline_out[:, target]
        else:
            baseline_score = baseline_out

    # Collect per-sample values for optimal scaling
    output_diffs = []
    attr_dot_perturbs = []

    for _ in range(n_perturb_samples):
        # Generate Gaussian perturbation
        perturbation = torch.randn_like(inputs) * perturb_scale

        # Perturbed prediction
        with torch.no_grad():
            perturbed_out = model_fn(inputs - perturbation)
            if perturbed_out.dim() > 1:
                perturbed_score = perturbed_out[:, target]
            else:
                perturbed_score = perturbed_out

        # Model output change: f(x) - f(x - perturbation)
        output_diff = (baseline_score - perturbed_score).sum().item()

        # Attribution-weighted perturbation (dot product)
        attr_dot_perturb = (attributions * perturbation).sum().item()

        output_diffs.append(output_diff)
        attr_dot_perturbs.append(attr_dot_perturb)

        # Clear memory
        del perturbation, perturbed_out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_diffs = np.array(output_diffs)
    attr_dot_perturbs = np.array(attr_dot_perturbs)

    if normalize:
        # Optimal scaling: beta = E[df * (attr.delta)] / E[(attr.delta)^2]
        denom = np.mean(attr_dot_perturbs ** 2)
        if denom > 0:
            beta = np.mean(output_diffs * attr_dot_perturbs) / denom
        else:
            beta = 1.0
        attr_dot_perturbs = attr_dot_perturbs * beta

    infidelity_score = np.mean((output_diffs - attr_dot_perturbs) ** 2)

    return infidelity_score


def compute_xai_metrics_segmentation(
    model,
    inputs,
    attributions,
    target,
    agg_wrapper,
    mask=None,
    aggregation="predicted",
    n_perturb_samples=10,
    perturb_radius=0.02,
    use_manual=True,
):
    """
    Compute XAI metrics for segmentation models using aggregation wrapper.

    Captum metrics require scalar or class-indexed outputs. This function
    wraps segmentation models to aggregate spatial outputs before computing
    infidelity and sensitivity metrics.

    Args:
        model: Segmentation model that outputs spatial predictions
        inputs: Input tensor (B, C, H, W, D) or (B, C, H, W)
        attributions: Attribution map matching input shape
        target: Target class index (int) for the aggregated output
        agg_wrapper: Aggregation wrapper function for the model
        mask: Optional mask tensor to focus aggregation on specific region
        aggregation: Aggregation strategy (unused if agg_wrapper provided)
        n_perturb_samples: Number of perturbation samples for sensitivity
        perturb_radius: Perturbation radius for sensitivity metric
        use_manual: If True, use memory-efficient manual implementations

    Returns:
        dict with 'infidelity' and 'sensitivity' scores
    """
    # Ensure attributions match input shape
    if attributions.shape != inputs.shape:
        raise ValueError(
            f"Attribution shape {attributions.shape} must match input shape {inputs.shape}"
        )

    if use_manual:
        # Memory-efficient manual implementations
        infid_score = compute_infidelity_manual(
            agg_wrapper,
            inputs,
            attributions,
            target=target,
            n_perturb_samples=n_perturb_samples
        )

        # Create attribution function for sensitivity
        saliency_fn = Saliency(agg_wrapper)

        def attr_fn(x, t):
            return saliency_fn.attribute(x, target=t, abs=True)

        sens_score = compute_sensitivity_manual(
            attr_fn,
            inputs,
            target,
            base_attributions=attributions,
            perturb_radius=perturb_radius,
            n_perturb_samples=n_perturb_samples
        )

        return {
            'infidelity': infid_score,
            'sensitivity': sens_score
        }
    else:
        # Original Captum implementations
        infid_score = infidelity(
            agg_wrapper,
            perturb_fn,
            inputs,
            attributions,
            target=target,
            normalize=True
        )

        saliency_fn = Saliency(agg_wrapper)

        sens_score = sensitivity_max(
            lambda x: saliency_fn.attribute(x, target=target, abs=True),
            inputs,
            perturb_radius=perturb_radius,
            n_perturb_samples=n_perturb_samples
        )

        return {
            'infidelity': infid_score.mean().item(),
            'sensitivity': sens_score.mean().item()
        }