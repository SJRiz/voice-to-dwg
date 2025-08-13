from fastapi import HTTPException
import speech_recognition as sr
import ezdxf
import json
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import Dict, Any, Union, Optional
import re
import os
import google.generativeai as genai
import uuid
import io
import time
from dotenv import load_dotenv

from file_cache import InMemoryFileCache

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

class VoiceToDWGProcessor:
    def __init__(self, cache_max_items: int = 50, cache_max_bytes: int = 50 * 1024 * 1024):
        self.recognizer = sr.Recognizer()
        self.file_cache = InMemoryFileCache(max_items=cache_max_items, max_bytes=cache_max_bytes)

    # audio_input can be:
    # - bytes (raw file bytes, e.g. UploadFile.read())
    # - a file-like object (has .read())
    # - a filesystem path (str)
    def transcribe_audio(self, audio_input: Union[str, bytes, io.IOBase]) -> str:
        """Convert audio to text using speech_recognition.
        Tries WAV parsing first (no ffmpeg). Falls back to pydub.from_file (requires ffmpeg).
        Raises HTTPException if ffmpeg is missing.
        """
        try:
            # Normalize input, get bytes or file-like
            if isinstance(audio_input, (bytes, bytearray)):
                bio = io.BytesIO(audio_input)
            elif hasattr(audio_input, "read"):
                audio_input.seek(0)
                bio = io.BytesIO(audio_input.read())
            elif isinstance(audio_input, str):
                # path on disk
                with open(audio_input, "rb") as f:
                    bio = io.BytesIO(f.read())
            else:
                raise ValueError("Unsupported audio_input type. Use bytes, file-like, or path string.")

            bio.seek(0)
            transcript_parts = []

            # Try fast path: if data looks like WAV, use sr.AudioFile directly (no ffmpeg)
            header = bio.read(12)
            bio.seek(0)
            is_wav = header[0:4] == b'RIFF'  # simple WAV check

            if is_wav:
                with sr.AudioFile(bio) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    return text.strip()

            # Otherwise, try pydub.from_file (needs ffmpeg). We split on silence for better accuracy.
            try:
                audio = AudioSegment.from_file(bio)   # may raise FileNotFoundError if ffmpeg not installed
            except FileNotFoundError as ff_err:

                # ffmpeg is missing
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Transcription failed: 'ffmpeg' not found on the server. "
                    )
                ) from ff_err

            chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
            for chunk in chunks:
                buf = io.BytesIO()
                chunk.export(buf, format="wav")
                buf.seek(0)
                with sr.AudioFile(buf) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data)
                    transcript_parts.append(text)

            return " ".join(transcript_parts).strip()

        except sr.UnknownValueError:
            raise HTTPException(status_code=400, detail="Transcription failed: could not understand audio")
        except sr.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Transcription failed (speech API error): {e}")
        except HTTPException:
            raise
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
        
        If dimensions or positions aren't specified, use reasonable defaults for the room type.
        Return only the JSON object, no other text.
        """
        
        try:
            response = model.generate_content(prompt)
            json_text = response.text.strip()
            if json_text.startswith('```json'):
                json_text = json_text[7:-3]
            elif json_text.startswith('```'):
                json_text = json_text[3:-3]
            return json.loads(json_text)
        except Exception:
            # fallback
            return self._fallback_parameter_extraction(transcript)

    def _fallback_parameter_extraction(self, transcript: str) -> Dict[str, Any]:
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

    def _parse_number(self, v):
        """Parse int/float/string like '12', '12.5', '12 ft', return float or None."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            # find first number in the string, allow decimals
            m = re.search(r'(-?\d+(\.\d+)?)', v)
            if m:
                try:
                    return float(m.group(1))
                except:
                    return None
        return None

    def _ensure_dimensions(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Ensure 'dimensions' exist and return tuple (length, width, unit).
        If missing or invalid, set reasonable defaults based on room_type.
        """
        # defaults by room type (length, width, unit)
        defaults = {
            "kitchen": (12.0, 10.0, "feet"),
            "bedroom": (15.0, 12.0, "feet"),
            "living room": (16.0, 14.0, "feet"),
            "living_room": (16.0, 14.0, "feet"),
            "office": (12.0, 10.0, "feet"),
            "bathroom": (8.0, 6.0, "feet"),
            "room": (10.0, 10.0, "feet")
        }

        room_type = (parameters.get("room_type") or "").lower() if parameters else ""
        if not room_type:
            room_type = "room"

        # access dimensions safely
        dims = (parameters or {}).get("dimensions") or {}
        length_raw = dims.get("length")
        width_raw = dims.get("width")
        unit = dims.get("unit") or "feet"

        length = self._parse_number(length_raw)
        width = self._parse_number(width_raw)

        if length is None or width is None:
            # fallback to defaults for detected room_type, or generic default
            chosen = defaults.get(room_type, defaults["room"])
            if length is None:
                length = float(chosen[0])
            if width is None:
                width = float(chosen[1])
            # prefer unit from dims if available else default choice
            unit = unit or chosen[2]

        # final sanity checks
        if length <= 0 or width <= 0:
            raise HTTPException(status_code=400, detail=f"Invalid room dimensions: length={length}, width={width}")

        return {"length": float(length), "width": float(width), "unit": str(unit)}

    def generate_dwg(self, parameters: Dict[str, Any]) -> str:
        """
        Generate a DXF (DWG-like) file in-memory and store it in the in-memory cache.
        Validates and normalizes dimensions before drawing.
        """
        try:
            # Validate / normalize dimensions (will raise HTTPException if invalid)
            dims = self._ensure_dimensions(parameters)

            length = float(dims['length'])
            width = float(dims['width'])

            # create dxf doc and draw
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()

            points = [
                (0, 0),
                (length, 0),
                (length, width),
                (0, width),
                (0, 0)
            ]
            msp.add_lwpolyline(points, close=True)

            room_label = (parameters.get('room_type') or 'Room').title()
            msp.add_text(
                f"{room_label}\n{length}' x {width}'",
                dxfattribs={
                    'height': 1,
                    'insert': (length/2, width/2),
                    'halign': 1,
                    'valign': 1
                }
            )

            for element in parameters.get('elements', []) if parameters else []:
                # ensure element size has numeric width to avoid None -> float error
                try:
                    self._add_element_to_drawing(msp, element, length, width)
                except Exception:
                    # skip bad element but don't crash entire generation
                    continue

            # produce unique filename (key)
            file_id = str(uuid.uuid4())[:8]
            filename = f"drawing_{file_id}.dxf"

            # write text and encode (ezdxf writes text)
            try:
                text_buf = io.StringIO()
                doc.write(text_buf)
                dxf_text = text_buf.getvalue()
                data = dxf_text.encode('utf-8')
            except Exception as e_text:
                # fallback using TextIOWrapper -> BytesIO
                try:
                    bin_buf = io.BytesIO()
                    text_wrapper = io.TextIOWrapper(bin_buf, encoding='utf-8', newline='')
                    doc.write(text_wrapper)
                    text_wrapper.flush()
                    bin_buf.seek(0)
                    data = bin_buf.read()
                except Exception as e_bin:
                    raise HTTPException(status_code=500,
                        detail=f"Failed to export DXF to memory (text and binary attempts failed): {e_text} | {e_bin}"
                    )

            metadata = {"filename": filename, "created_at": time.time(), "size": len(data)}
            self.file_cache.set(filename, data, metadata=metadata)
            return filename

        except HTTPException:
            # propagate HTTPExceptions (useful for returning 400 to client)
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DWG generation failed: {str(e)}")

    def _add_element_to_drawing(self, msp, element, room_length, room_width):
        element_type = element['type']
        position = element.get('position', 'north')
        size = element.get('size', {'width': 3, 'height': 1})
        
        room_length = float(room_length)
        room_width = float(room_width)
        element_width = float(size['width'])
        
        if element_type == 'door':
            if position in ['east', 'right']:
                door_y = room_width / 2 - element_width / 2
                msp.add_line((room_length, door_y), (room_length, door_y + element_width))
                msp.add_arc(
                    center=(room_length, door_y),
                    radius=element_width,
                    start_angle=180,
                    end_angle=270
                )
                msp.add_text("DOOR", dxfattribs={'height': 0.5, 'insert': (room_length - 1, door_y + element_width/2)})
            elif position in ['west', 'left']:
                door_y = room_width / 2 - element_width / 2
                msp.add_line((0, door_y), (0, door_y + element_width))
                msp.add_arc(center=(0, door_y + element_width), radius=element_width, start_angle=0, end_angle=90)
                msp.add_text("DOOR", dxfattribs={'height': 0.5, 'insert': (1, door_y + element_width/2)})
            elif position in ['north', 'front']:
                door_x = room_length / 2 - element_width / 2
                msp.add_line((door_x, room_width), (door_x + element_width, room_width))
                msp.add_arc(center=(door_x, room_width), radius=element_width, start_angle=270, end_angle=360)
                msp.add_text("DOOR", dxfattribs={'height': 0.5, 'insert': (door_x + element_width/2, room_width - 1)})
            else:  # south/back
                door_x = room_length / 2 - element_width / 2
                msp.add_line((door_x, 0), (door_x + element_width, 0))
                msp.add_arc(center=(door_x + element_width, 0), radius=element_width, start_angle=90, end_angle=180)
                msp.add_text("DOOR", dxfattribs={'height': 0.5, 'insert': (door_x + element_width/2, 1)})
        
        elif element_type == 'window':
            if position in ['north', 'front']:
                window_x = room_length / 2 - element_width / 2
                msp.add_line((window_x, room_width), (window_x + element_width, room_width))
                msp.add_line((window_x, room_width - 0.2), (window_x + element_width, room_width - 0.2))
                msp.add_text("WINDOW", dxfattribs={'height': 0.3, 'insert': (window_x + element_width/2, room_width - 0.5)})
            elif position in ['south', 'back']:
                window_x = room_length / 2 - element_width / 2
                msp.add_line((window_x, 0), (window_x + element_width, 0))
                msp.add_line((window_x, 0.2), (window_x + element_width, 0.2))
                msp.add_text("WINDOW", dxfattribs={'height': 0.3, 'insert': (window_x + element_width/2, 0.5)})
            elif position in ['east', 'right']:
                window_y = room_width / 2 - element_width / 2
                msp.add_line((room_length, window_y), (room_length, window_y + element_width))
                msp.add_line((room_length - 0.2, window_y), (room_length - 0.2, window_y + element_width))
                msp.add_text("WINDOW", dxfattribs={'height': 0.3, 'insert': (room_length - 0.5, window_y + element_width/2), 'rotation': 90})
            elif position in ['west', 'left']:
                window_y = room_width / 2 - element_width / 2
                msp.add_line((0, window_y), (0, window_y + element_width))
                msp.add_line((0.2, window_y), (0.2, window_y + element_width))
                msp.add_text("WINDOW", dxfattribs={'height': 0.3, 'insert': (0.5, window_y + element_width/2), 'rotation': 90})

    # retrieval helpers
    def get_file_bytes(self, filename: str) -> Optional[bytes]:
        return self.file_cache.get(filename)

    def get_file_metadata(self, filename: str) -> Optional[Dict]:
        return self.file_cache.get_metadata(filename)

    def list_files(self):
        return self.file_cache.list_keys()

    def delete_file(self, filename: str) -> bool:
        return self.file_cache.delete(filename)