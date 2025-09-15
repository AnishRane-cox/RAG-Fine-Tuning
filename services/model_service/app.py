from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys, os
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../scripts")))
from inference_adapter import load_model_and_adapter, generate_answer
load_dotenv()
BASE_MODEL = os.getenv("BASE_MODEL", "distilgpt2")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "./inference_adapter")

tokenizer, model = load_model_and_adapter(BASE_MODEL, ADAPTER_PATH)

app = FastAPI(title="LoRA Model Service")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64

class AdapterRequest(BaseModel):
    adapter_path: str

@app.get("/health")
def health():
    return {"status": "ok", "base_model": BASE_MODEL}

@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        ans = generate_answer(tokenizer, model, req.prompt, req.max_new_tokens)
        return {"answer": ans}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_adapter")
def load_adapter(req: AdapterRequest):
    global tokenizer, model
    try:
        tokenizer, model = load_model_and_adapter(BASE_MODEL, req.adapter_path)
        return {"status": "ok", "adapter": req.adapter_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))