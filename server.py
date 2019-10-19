from sentence_transformers import SentenceTransformer
import scipy
import json
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

model = SentenceTransformer('m')

from fastapi import FastAPI

app = FastAPI()

templates = Jinja2Templates(directory=".")
@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

app.mount("/includes", StaticFiles(directory="./includes"))

@app.get("/api")
def infer(a: str = None,one:str=None,two:str=None,three:str=None):
    anchor = [a]
    anchor_embedding = model.encode(anchor)
    responses = [one,two,three]
    responses_embedding = model.encode(responses)
    distances = scipy.spatial.distance.cdist(anchor_embedding, responses_embedding, "cosine")[0]
    return {"one":distances[0],"two":distances[1],"three":distances[2]}