#!/usr/bin/env python3
"""
Wayfarer API Server
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import time

app = FastAPI(title="Wayfarer-2-12B API", version="1.0.0")

# Global model instance
model = None
tokenizer = None

class ChatRequest(BaseModel):
    message: str
    max_length: int = 512
    temperature: float = 0.7

class ChatResponse(BaseModel):
    response: str
    generation_time: float
    tokens_generated: int

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    
    print("🚀 Loading Wayfarer model...")
    
    model_path = "./wayfarer-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate response to user message"""
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format prompt
        prompt = f"<|im_start|>user\n{request.message}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        return ChatResponse(
            response=response,
            generation_time=generation_time,
            tokens_generated=len(outputs[0]) - len(inputs.input_ids[0])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
