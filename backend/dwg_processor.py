from fastapi import HTTPException
import speech_recognition as sr
import ezdxf
import json
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import Dict, Any
import re
import os
import google.generativeai as genai
import tempfile
from dotenv import load_dotenv
import uuid
from pathlib import Path

load_dotenv()

# Create uploads directory for storing generated files
UPLOAD_DIR = Path("generated_files")
UPLOAD_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

class VoiceToDWGProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def transcribe_audio(self, audio_file_path: str) -> str:
        """Convert audio to text using speech recognition"""
        try:
            # Convert to WAV if needed
            if not audio_file_path.endswith('.wav'):
                audio = AudioSegment.from_file(audio_file_path)
                wav_path = audio_file_path.replace(audio_file_path.split('.')[-1], 'wav')
                audio.export(wav_path, format="wav")
                audio_file_path = wav_path
            
            # Split audio on silence for better recognition
            audio = AudioSegment.from_wav(audio_file_path)
            chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
            
            transcript = ""
            for chunk in chunks:
                chunk_path = tempfile.mktemp(suffix='.wav')
                chunk.export(chunk_path, format="wav")
                
                with sr.AudioFile(chunk_path) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    transcript += text + " "
                
                os.remove(chunk_path)
            
            return transcript.strip()
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Transcription failed: {str(e)}")
    
    def extract_drawing_parameters(self, transcript: str) -> Dict[str, Any]:
        """Use Gemini to extract structured drawing parameters from transcript"""
        prompt = f"""
        From this transcript: "{transcript}"
        
        Extract drawing parameters and return a JSON object with the following structure:
        {{
            "room_type": "kitchen/bedroom/living_room/office/bathroom",
            "dimensions": {{
                "length": number,
                "width": number,
                "unit": "feet/meters"
            }},
            "elements": [
                {{
                    "type": "door/window/wall/fixture",
                    "position": "north/south/east/west/front/back/left/right",
                    "size": {{
                        "width": number,
                        "height": number
                    }}
                }}
            ],
            "additional_notes": "any other specifications"
        }}
        
        If dimensions aren't specified, use reasonable defaults for the room type.
        Return only the JSON object, no other text.
        """
        
        try:
            response = model.generate_content(prompt)
            # Clean the response to get just the JSON
            json_text = response.text.strip()
            if json_text.startswith('```json'):
                json_text = json_text[7:-3]
            elif json_text.startswith('```'):
                json_text = json_text[3:-3]
            
            return json.loads(json_text)
        except Exception as e:
            # Fallback parsing if Gemini fails
            return self._fallback_parameter_extraction(transcript)
    
    def _fallback_parameter_extraction(self, transcript: str) -> Dict[str, Any]:
        """Fallback method to extract basic parameters using regex"""
        dimensions = re.findall(r'(\d+)x(\d+)', transcript.lower())
        room_types = ['kitchen', 'bedroom', 'living room', 'office', 'bathroom']
        
        room_type = "room"
        for rt in room_types:
            if rt in transcript.lower():
                room_type = rt
                break
        
        length, width = (10, 10)  # Default
        if dimensions:
            length, width = int(dimensions[0][0]), int(dimensions[0][1])
        
        elements = []
        if 'door' in transcript.lower():
            position = 'east'
            if 'right' in transcript.lower(): position = 'east'
            elif 'left' in transcript.lower(): position = 'west'
            elif 'front' in transcript.lower(): position = 'north'
            elif 'back' in transcript.lower(): position = 'south'
            
            elements.append({
                "type": "door",
                "position": position,
                "size": {"width": 3, "height": 7}
            })
        
        if 'window' in transcript.lower():
            elements.append({
                "type": "window",
                "position": "north",
                "size": {"width": 4, "height": 3}
            })
        
        return {
            "room_type": room_type,
            "dimensions": {"length": length, "width": width, "unit": "feet"},
            "elements": elements,
            "additional_notes": transcript
        }
    
    def generate_dwg(self, parameters: Dict[str, Any]) -> str:
        """Generate DWG file using ezdxf based on extracted parameters"""
        try:
            # Create new DXF document
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Get room dimensions
            length = float(parameters['dimensions']['length'])
            width = float(parameters['dimensions']['width'])
            
            # Draw room outline (rectangle)
            points = [
                (0, 0),
                (length, 0),
                (length, width),
                (0, width),
                (0, 0)
            ]
            msp.add_lwpolyline(points, close=True)
            
            # Add room label
            msp.add_text(
                f"{parameters['room_type'].title()}\n{length}' x {width}'",
                dxfattribs={
                    'height': 1,
                    'insert': (length/2, width/2),
                    'halign': 1,  # Center horizontal
                    'valign': 1   # Center vertical
                }
            )
            
            # Add elements (doors, windows, etc.)
            for element in parameters.get('elements', []):
                self._add_element_to_drawing(msp, element, length, width)
            
            # Generate unique filename
            file_id = str(uuid.uuid4())[:8]
            filename = f"drawing_{file_id}.dxf"
            file_path = UPLOAD_DIR / filename
            
            # Save file
            doc.saveas(str(file_path))
            
            return filename  # Return just the filename, not full path
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DWG generation failed: {str(e)}")
    
    def _add_element_to_drawing(self, msp, element, room_length, room_width):
        """Add doors, windows, and other elements to the drawing"""
        element_type = element['type']
        position = element['position']
        size = element.get('size', {'width': 3, 'height': 1})
        
        # Ensure dimensions are float
        room_length = float(room_length)
        room_width = float(room_width)
        element_width = float(size['width'])
        
        if element_type == 'door':
            # Add door based on position
            if position in ['east', 'right']:
                # Door on right wall
                door_y = room_width / 2 - element_width / 2
                # Add door opening (gap in wall)
                msp.add_line((room_length, door_y), (room_length, door_y + element_width))
                # Add door arc (swing)
                msp.add_arc(
                    center=(room_length, door_y),
                    radius=element_width,
                    start_angle=180,
                    end_angle=270
                )
                # Add door label
                msp.add_text(
                    "DOOR",
                    dxfattribs={
                        'height': 0.5,
                        'insert': (room_length - 1, door_y + element_width/2)
                    }
                )
            elif position in ['west', 'left']:
                # Door on left wall
                door_y = room_width / 2 - element_width / 2
                msp.add_line((0, door_y), (0, door_y + element_width))
                msp.add_arc(
                    center=(0, door_y + element_width),
                    radius=element_width,
                    start_angle=0,
                    end_angle=90
                )
                msp.add_text(
                    "DOOR",
                    dxfattribs={
                        'height': 0.5,
                        'insert': (1, door_y + element_width/2)
                    }
                )
            elif position in ['north', 'front']:
                # Door on front wall
                door_x = room_length / 2 - element_width / 2
                msp.add_line((door_x, room_width), (door_x + element_width, room_width))
                msp.add_arc(
                    center=(door_x, room_width),
                    radius=element_width,
                    start_angle=270,
                    end_angle=360
                )
                msp.add_text(
                    "DOOR",
                    dxfattribs={
                        'height': 0.5,
                        'insert': (door_x + element_width/2, room_width - 1)
                    }
                )
            else:  # south/back
                door_x = room_length / 2 - element_width / 2
                msp.add_line((door_x, 0), (door_x + element_width, 0))
                msp.add_arc(
                    center=(door_x + element_width, 0),
                    radius=element_width,
                    start_angle=90,
                    end_angle=180
                )
                msp.add_text(
                    "DOOR",
                    dxfattribs={
                        'height': 0.5,
                        'insert': (door_x + element_width/2, 1)
                    }
                )
        
        elif element_type == 'window':
            # Add window based on position
            if position in ['north', 'front']:
                window_x = room_length / 2 - element_width / 2
                # Double line for window
                msp.add_line((window_x, room_width), (window_x + element_width, room_width))
                msp.add_line((window_x, room_width - 0.2), (window_x + element_width, room_width - 0.2))
                # Window label
                msp.add_text(
                    "WINDOW",
                    dxfattribs={
                        'height': 0.3,
                        'insert': (window_x + element_width/2, room_width - 0.5)
                    }
                )
            elif position in ['south', 'back']:
                window_x = room_length / 2 - element_width / 2
                msp.add_line((window_x, 0), (window_x + element_width, 0))
                msp.add_line((window_x, 0.2), (window_x + element_width, 0.2))
                msp.add_text(
                    "WINDOW",
                    dxfattribs={
                        'height': 0.3,
                        'insert': (window_x + element_width/2, 0.5)
                    }
                )
            elif position in ['east', 'right']:
                window_y = room_width / 2 - element_width / 2
                msp.add_line((room_length, window_y), (room_length, window_y + element_width))
                msp.add_line((room_length - 0.2, window_y), (room_length - 0.2, window_y + element_width))
                msp.add_text(
                    "WINDOW",
                    dxfattribs={
                        'height': 0.3,
                        'insert': (room_length - 0.5, window_y + element_width/2),
                        'rotation': 90
                    }
                )
            elif position in ['west', 'left']:
                window_y = room_width / 2 - element_width / 2
                msp.add_line((0, window_y), (0, window_y + element_width))
                msp.add_line((0.2, window_y), (0.2, window_y + element_width))
                msp.add_text(
                    "WINDOW",
                    dxfattribs={
                        'height': 0.3,
                        'insert': (0.5, window_y + element_width/2),
                        'rotation': 90
                    }
                )