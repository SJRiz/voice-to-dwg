from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
from typing import Dict
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
processor = VoiceToDWGProcessor(cache_max_items=50, cache_max_bytes=50 * 1024 * 1024)


@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Endpoint to transcribe audio file"""
    try:
        audio_bytes = await audio_file.read()
        transcript = processor.transcribe_audio(audio_bytes)
        return {"transcript": transcript}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-parameters")
async def extract_parameters(data: Dict):
    """Extract drawing parameters from transcript"""
    transcript = data.get("transcript", "")
    if not transcript:
        raise HTTPException(status_code=400, detail="No transcript provided")
    parameters = processor.extract_drawing_parameters(transcript)
    return {"parameters": parameters}


@app.post("/generate-dwg")
async def generate_dwg(data: Dict):
    """Generate DWG file from parameters and store it in-memory cache."""
    parameters = data.get("parameters", {})
    if not parameters:
        raise HTTPException(status_code=400, detail="No parameters provided")
    filename = processor.generate_dwg(parameters)
    return {"dwg_filename": filename, "download_url": f"/download-dwg/{filename}"}


@app.get("/download-dwg/{filename}")
async def download_dwg(filename: str):
    """Stream DXF file from in-memory cache"""
    data = processor.get_file_bytes(filename)
    if not data:
        raise HTTPException(status_code=404, detail="File not found")
    buf = io.BytesIO(data)
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(buf, media_type="application/dxf", headers=headers)


# Helper
@app.get("/list-dwgs")
def list_dwgs():
    """List cached DWG filenames (most recent first)."""
    return {"files": processor.list_files()}


@app.delete("/delete-dwg/{filename}")
def delete_dwg(filename: str):
    ok = processor.delete_file(filename)
    if not ok:
        raise HTTPException(status_code=404, detail="File not found")
    return {"deleted": filename}


@app.post("/voice-to-dwg")
async def voice_to_dwg_complete(audio_file: UploadFile = File(...)):
    """voice -> transcript -> parameters -> DWG"""
    try:
        audio_bytes = await audio_file.read()
        transcript = processor.transcribe_audio(audio_bytes)
        parameters = processor.extract_drawing_parameters(transcript)
        filename = processor.generate_dwg(parameters)
        return {
            "transcript": transcript,
            "parameters": parameters,
            "dwg_filename": filename,
            "download_url": f"/download-dwg/{filename}"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)