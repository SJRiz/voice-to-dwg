from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tempfile
import os

from dwg_processor import VoiceToDWGProcessor

app = FastAPI(title="Voice-to-DWG API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor
processor = VoiceToDWGProcessor()

@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Endpoint to transcribe audio file"""
    
    # Save uploaded file
    temp_audio_path = tempfile.mktemp(suffix=f".{audio_file.filename.split('.')[-1]}")
    with open(temp_audio_path, "wb") as buffer:
        content = await audio_file.read()
        buffer.write(content)
    
    try:
        transcript = processor.transcribe_audio(temp_audio_path)
        return {"transcript": transcript}
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.post("/extract-parameters")
async def extract_parameters(data: dict):
    """Extract drawing parameters from transcript"""
    transcript = data.get("transcript", "")
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided")
    
    parameters = processor.extract_drawing_parameters(transcript)
    return {"parameters": parameters}

@app.post("/generate-dwg")
async def generate_dwg(data: dict):
    """Generate DWG file from parameters"""
    parameters = data.get("parameters", {})
    if not parameters:
        raise HTTPException(status_code=400, detail="No parameters provided")
    
    dwg_filename = processor.generate_dwg(parameters)
    return {"dwg_filename": dwg_filename, "message": "DWG generated successfully"}

@app.get("/download-dwg/{file_path}")
async def download_dwg(file_path: str):
    """Download generated DWG file"""
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/octet-stream', filename='drawing.dxf')
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.post("/voice-to-dwg")
async def voice_to_dwg_complete(audio_file: UploadFile = File(...)):
    """voice -> transcript -> parameters -> DWG"""
    # Save uploaded file
    temp_audio_path = tempfile.mktemp(suffix=f".{audio_file.filename.split('.')[-1]}")
    with open(temp_audio_path, "wb") as buffer:
        content = await audio_file.read()
        buffer.write(content)
    
    try:
        # Transcribe
        transcript = processor.transcribe_audio(temp_audio_path)
        
        # Extract parameters
        parameters = processor.extract_drawing_parameters(transcript)
        
        # Generate DWG
        dwg_path = processor.generate_dwg(parameters)
        
        return {
            "transcript": transcript,
            "parameters": parameters,
            "dwg_path": dwg_path,
            "download_url": f"/download-dwg/{dwg_path}"
        }
    
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)