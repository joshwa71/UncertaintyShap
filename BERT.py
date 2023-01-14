import shap
import transformers
import nlp
import torch
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import webbrowser

# load a BERT sentiment analysis model
tokenizer = transformers.DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
).cuda()

def entropy(x):
    _x = x
    logp = np.log(_x)
    plogp = np.multiply(_x, logp)
    out = np.sum(plogp, axis=1)
    return -out

def f(x):
    tv = torch.tensor([tokenizer.encode(v, pad_to_max_length=True, max_length=500) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = entropy(scores)
    #val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

imdb_train = nlp.load_dataset("imdb")["train"]

# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)

# explain the model's predictions on IMDB reviews

shap_values = explainer(imdb_train[:10])

for i in range(10):
    file = open(str(i) + '.html','w')
    file.write(shap.plots.text(shap_values[1], display=False))
    file.close