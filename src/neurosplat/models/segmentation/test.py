import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForSemanticSegmentation



model_v = "Intel/dpt-large-ade"
img = Image.open("/home/ram/Downloads/360_v2/counter/images_4/DSCF5859.JPG")
f_extructer = DPTImageProcessor.from_pretrained(model_v)
model = DPTForSemanticSegmentation.from_pretrained(model_v)
model_subs_ = [model.dpt, model.neck, model.head]
model_cfg_ = model.config


features = f_extructer(img, return_tensors="pt").pixel_values
print(features.size())
dpt_output = model.dpt(features, output_hidden_states=True)
hidden_states = dpt_output.hidden_states
hidden_states = [
    feature
    for (idx, feature) in enumerate(hidden_states[1:])
    if idx in model_cfg_.backbone_out_indices
]
# print(model_cfg_)
print(f"HIDDEN STATES AFTER DPT BACKBONE --- {len(hidden_states)}", "-" * 12)
for state in hidden_states:
    print(state.size())
print(f"HIDDEN STATES AFTER DPT BACKBONE --- {len(hidden_states)}", "-" * 12)

hidden_states = model.neck(hidden_states)
print(f"HIDDEN STATES AFTER DPT NECK --- {len(hidden_states)}", "-" * 12)
for state in hidden_states:
    print(state.size())
print(f"HIDDEN STATES AFTER DPT NECK --- {len(hidden_states)}", "-" * 12)

print(type(hidden_states), type(hidden_states[0]))
logits = model.head(hidden_states)
logits = torch.argmax(logits, dim=1).squeeze().numpy().astype("uint8")

colors = [0,0,0,120,120,120,180,120,120,6,230,230,80,50,50,4,200,3,120,120,80,140,140,140,204,5,255,230,230,230,4,250,7,224,5,255,235,255,7,150,5,61,120,120,70,8,255,51,255,6,82,143,255,140,204,255,4,255,51,7,204,70,3,0,102,200,61,230,250,255,6,51,11,102,255,255,7,71,255,9,224,9,7,230,220,220,220,255,9,92,112,9,255,8,255,214,7,255,224,255,184,6,10,255,71,255,41,10,7,255,255,224,255,8,102,8,255,255,61,6,255,194,7,255,122,8,0,255,20,255,8,41,255,5,153,6,51,255,235,12,255,160,150,20,0,163,255,140,140,140,250,10,15,20,255,0,31,255,0,255,31,0,255,224,0,153,255,0,0,0,255,255,71,0,0,235,255,0,173,255,31,0,255,11,200,200,255,82,0,0,255,245,0,61,255,0,255,112,0,255,133,255,0,0,255,163,0,255,102,0,194,255,0,0,143,255,51,255,0,0,82,255,0,255,41,0,255,173,10,0,255,173,255,0,0,255,153,255,92,0,255,0,255,255,0,245,255,0,102,255,173,0,255,0,20,255,184,184,0,31,255,0,255,61,0,71,255,255,0,204,0,255,194,0,255,82,0,10,255,0,112,255,51,0,255,0,194,255,0,122,255,0,255,163,255,153,0,0,255,10,255,112,0,143,255,0,82,0,255,163,255,0,255,235,0,8,184,170,133,0,255,0,255,92,184,0,255,255,0,31,0,184,255,0,214,255,255,0,112,92,255,0,0,224,255,112,224,255,70,184,160,163,0,255,153,0,255,71,255,0,255,0,163,255,204,0,255,0,143,0,255,235,133,255,0,255,0,235,245,0,255,255,0,122,255,245,0,10,190,212,214,255,0,0,204,255,20,0,255,255,255,0,0,153,255,0,41,255,0,255,204,41,0,255,41,255,0,173,0,255,0,245,255,71,0,255,122,0,255,0,255,184,0,92,255,184,255,0,0,133,255,255,214,0,25,194,194,102,255,0,92,0,255]
print(logits.shape)
mask = Image.fromarray(logits)
mask.putpalette(colors)
mask = mask.convert("RGB")
img = img.resize((480, 480))
print(np.asarray(mask).shape, np.asarray(img).shape)
rgb_result = np.asarray(Image.blend(img, mask, alpha=0.5))
rgb_result = (rgb_result / 255.0)

print(rgb_result.shape)
import matplotlib.pyplot as plt
_, axis = plt.subplots()
axis.imshow(rgb_result)
plt.show()


# print(torch.max(logits))
# print(logits.size())


    
   


