
from trainer import LitModel
import torch 
from shared_modules.data_module import DataModule
from shared_modules.utils import load_config
from monai.transforms import CenterSpatialCrop
from tqdm import tqdm
from shared_modules.plotting import slice_comparison_multi
from monai.transforms import ScaleIntensityRangePercentiles
from captum.attr import Occlusion
from captum.attr import Saliency
import matplotlib.pyplot as plt


# Settings:
SAVE_PREDS=False
SAVE_PROB_MAPS=False
dataset="picai" 
label_key = "pca"
config = load_config("config.yaml")
gpu = 0
config.gpus = [gpu]
config.cache_rate = 1.0
config.transforms.label_keys = ["pca", "prostate_pred", "zones"]
checkpoint_path = "/cluster/home/bragehk/U-MambaMTL-XAI/gc_algorithms/base_container/models/umamba_mtl/weights/f0.ckpt"
model = LitModel.load_from_checkpoint(checkpoint_path, config=config)

# disable randomness, dropout, etc...
model = model.eval()
model.to(gpu)

dm = DataModule(
    config=config,
)

dm.setup("training")
dl = dm.train_dataloader() #TODO: The image should not be an image that the model has trained on
# dm.setup("test")
# dl = dm.test_dataloader()


def agg_segmentation_wrapper(inp):
    # Get the logits from the model
    model_out = model(inp)  

    # Find the predicted class for each pixel 
    # (argmax along the channel dimension)
    out_max = model_out.argmax(dim=1, keepdim=True)

    # Create a binary mask with 1 for the predicted class and 0 otherwise
    # selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)
    selected_inds = torch.zeros_like(model_out).scatter_(1, out_max, 1)

    # Multiply the logits with the mask and sum across spatial dimensions
    # Sum over x, y, and z
    aggregated_logits = (model_out * selected_inds).sum(dim=(2, 3, 4))  
    # aggregated_logits = (model_out).sum(dim=(2, 3, 4))  

    return aggregated_logits


occlusion = Occlusion(agg_segmentation_wrapper)
attribute_fn = Saliency(agg_segmentation_wrapper)

sliding_window_shapes = ( 1, 32, 32, 5)  # Occlude one channel at a time
strides = ( 1, 16, 16, 5)                # Stride within each channel 
baselines = 0                          # Value to fill the occluded areas (0 is common)

tz = 0
pz = 0

i = 0

logits = None

for batch in tqdm(dl):
    if batch["pca"].max() == 0:
        print("B")
        continue
    
    i = i +1
    if i < 42: # A way to select new positive cases for plotting
        print("S")
        continue
    
    x = batch["image"].to(gpu)
    logits = model(x)
    if (torch.sigmoid(logits) > 0.5).int()[0][1][None].to("cpu")[0].max() == 1:
        # attention_map = attribute_fn.attribute(x, target=1, abs=False)
        attention_map = attribute_fn.attribute(x, target=1, abs=True)
        # occlusion_map = occlusion.attribute(x, sliding_window_shapes, strides, baselines, target=1)
    else:
        print("False negative..")
        print((torch.sigmoid(logits) > 0.5).int()[0][1][None].to("cpu")[0,1].max().item())
        
    print(i)
    pca_in_pz = (batch["pca"] * batch["zones"] == 1).sum()
    pca_in_tz = (batch["pca"] * batch["zones"] == 2).sum()
    
    break

def normalize_and_clamp(x):
    for i in range(3):
        # print((x[i] > 0).sum())
        # print((x[i] < 0).sum())
        # x[i] = torch.clamp(x[i] + 0.5 - x[i].mean(), min=0.001, max=0.999)
        # x[i] = torch.clamp(x[i] + 0.5 - x[i].mean(), min=-0.999, max=0.999)
        x[i] = x[i] + 0.5 - x[i].mean()
        # if i == 2:
        #     x[i] = x[i] *2
        # x[i] = torch.clamp(x[i] - x[i].mean(), min=0.001, max=0.999)
        # print((x[i] > 0).sum())
        # print((x[i] < 0).sum())
    return x


from monai.transforms import NormalizeIntensity

img = batch["image"][0]
gt = batch["pca"][0]
pred = (torch.sigmoid(logits) > 0.5).int()[0][1][None].to("cpu")
logit = logits[0][1][None].to("cpu")

activation = attention_map[0].to("cpu")

img = CenterSpatialCrop(roi_size=[128,128,20])(img)
activation = CenterSpatialCrop(roi_size=[128,128,20])(activation)

#occlusion = CenterSpatialCrop(roi_size=[128,128,20])(occlusion_map[0].to("cpu"))
occlusion = CenterSpatialCrop(roi_size=[128,128,20])(attention_map[0].to("cpu"))

occ = ScaleIntensityRangePercentiles(lower=.1, upper=99.9, b_min=-1, b_max=1, clip=True)(occlusion)
# occ = ScaleIntensityRangePercentiles(lower=.0, upper=100, b_min=-1, b_max=1)(occlusion)
acc = ScaleIntensityRangePercentiles(lower=.1, upper=99.9, b_min=-1, b_max=1, clip=True)(activation)
# acc = ScaleIntensityRangePercentiles(lower=.0, upper=100, b_min=0, b_max=1)(activation)
print(acc.max())
print(acc.min())
# acc = normalize_and_clamp(acc)
print(acc.max())
print(acc.min())
occ = normalize_and_clamp(occ) 
# occ = occlusion

gt = CenterSpatialCrop(roi_size=[128,128,20])(gt)
pred = CenterSpatialCrop(roi_size=[128,128,20])(pred)
logit = CenterSpatialCrop(roi_size=[128,128,20])(logit)

argmax_pred = logits.argmax(dim=1, keepdim=True)[0].to("cpu")
argmax_pred = CenterSpatialCrop(roi_size=[128,128,20])(argmax_pred)

print(f"More pca in pz: {pca_in_pz > pca_in_tz}")
print(f"pca in tz: {pca_in_tz}")
print(f"pca in pz: {pca_in_pz}")
case_id = batch["image"].meta["filename_or_obj"][0].split("/")[-1]
print(f"Case ID: {case_id}")
confidence = round(torch.sigmoid(logits)[0,1].max().item() * 100, 2)
print(f"PCa confidence: {confidence}%")

# slice_comparison_multi(image=ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1)(img), labels=[gt,pred, acc, occ, logit ], titles=["Ground Truth","Prediction", "Saliency", "Occlusion", "logits"])
slice_comparison_multi(image=ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1)(img), labels=[gt,pred, acc, occ ], titles=["Ground Truth","Prediction", "Saliency", "Occlusion"])


print([acc[i].max() for i in range(3)])
print([acc[i].min() for i in range(3)])


# plt.imshow(activation[1,:,:,5], vmin=0, vmax=1)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white","white", "green"]
        )

fig, ax = plt.subplots(1,3)
for i in range(3):
    ax[i].imshow(acc[i,:,:,7], vmin=-.5, vmax=.5, cmap=cmap)


i = 2
# normalized = torch.clamp(acc[i] + 0.5 - acc[i].mean(), min=0, max=1)
print(acc[i].mean())
print(acc[i].max())
print(acc[i].min())
# (acc[i] + 0.5 - acc[i].mean()).mean()



from captum.attr import visualization as viz

for i in range(3):
    orig_image = img[i,:,:,7][None].permute(1, 2, 0)
    attr = activation[i,:,:,7][None].permute(1, 2, 0)
    viz.visualize_image_attr_multiple(attr=attr, original_image=orig_image, methods=["original_image", "blended_heat_map", "blended_heat_map"], signs=["all", "absolute_value", "all"], show_colorbar=True, use_pyplot=True)

orig_image = ScaleIntensityRangePercentiles(lower=0, upper=100, b_min=0, b_max=1)(img)[:,:,:,7].permute(1, 2, 0)
attr = occ_neg[:,:,:,7].permute(1, 2, 0)
viz.visualize_image_attr_multiple(attr=attr, original_image=orig_image, methods=["original_image", "blended_heat_map","blended_heat_map"], signs=["all", "all", "negative"], show_colorbar=True, cmap="coolwarm")

for i in range(3):
    orig_image = img[i,:,:,15][None].permute(1, 2, 0)
    attr = occlusion[i,:,:,15][None].permute(1, 2, 0)
    viz.visualize_image_attr_multiple(attr=attr, original_image=orig_image, methods=["original_image", "blended_heat_map","blended_heat_map"], signs=["all", "all", "negative"], show_colorbar=True)
