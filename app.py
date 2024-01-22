from fastai.vision.all import *
from pathlib import Path
import gradio as gr

def is_chicken(x): return x[0].isupper()

img = PILImage.create('chicken.jpeg')

img.thumbnail((128,128))

learn = load_learner('model.pkl')

learn.predict(img)

learn.export('model.pkl')

def classify_image(inp):
    img = PILImage.create(inp)
    img.thumbnail((128,128))
    pred,pred_idx,probs = learn.predict(img)
    if pred == "duck":
        result = f"The image is a duck with probability {probs[pred_idx]:.4f}"
    else:
        result = f"The image is look like a chicken {probs[pred_idx]:.4f}"
    return result

image = gr.Image()
label = gr.Label()
examples = ["chicken.jpeg", "duck.jpeg", "03Chicken.jpeg"]

gr.Interface(fn=classify_image, inputs="image", outputs="text", examples=examples).launch()