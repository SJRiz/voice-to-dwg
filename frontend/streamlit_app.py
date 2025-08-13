import streamlit as st
import requests
import io
import os
from dotenv import load_dotenv
from audio_recorder_streamlit import audio_recorder

load_dotenv()

# Configure page
st.set_page_config(
    page_title="Voice-to-DWG Assistant",
    page_icon="🎤",
    layout="wide"
)

# FastAPI backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def main():
    st.title("🎤 Voice-to-DWG Conversational Design Assistant 🎤")
    st.markdown("---")
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("How to Use")
        st.markdown("""
        1. **Record** your voice command
        2. **Review** the transcript
        3. **Check** extracted parameters
        4. **Download** your DWG file
        
        ### Example:
        - "Draw a 12x10 kitchen with a door on the right"
        - "Create a 15x12 bedroom with two windows"
        - "Make a 10x8 bathroom with a door on the left"
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Voice Input")
        
        # Audio recorder
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#2ecc71",
            icon_name="microphone",
            icon_size="2x"
        )
        
        # File upload as alternative
        st.markdown("**Or upload an audio file:**")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'm4a'],
            help="Upload a voice recording with your design instructions"
        )
        
        # Process button
        if st.button("Process Voice Command", type="primary", use_container_width=True):
            if audio_bytes or uploaded_file:
                process_voice_command(audio_bytes, uploaded_file, BACKEND_URL)
            else:
                st.error("Please record audio or upload a file first!")
    
    with col2:
        st.header("Results")
        
        # Display results from session state
        if 'transcript' in st.session_state:
            st.subheader("Transcript")
            st.text_area("What you said:", st.session_state.transcript, height=100, disabled=True)
        
        if 'parameters' in st.session_state:
            st.subheader("Extracted Parameters")
            st.json(st.session_state.parameters)
        
        if 'dwg_ready' in st.session_state and st.session_state.dwg_ready:
            st.subheader("📁 Download")
            # show download button directly using download_url or fetch on click
            if st.button("💾 Download DXF File", type="secondary", use_container_width=True):
                # use the stored filename
                filename_key = st.session_state.get("dwg_filename") or st.session_state.get("dwg_path")
                if not filename_key:
                    st.error("No DWG filename available.")
                else:
                    download_dwg_file(filename_key, BACKEND_URL)

def process_voice_command(audio_bytes, uploaded_file, backend_url):
    """Process the voice command through the complete pipeline without disk I/O."""
    
    with st.spinner("Processing your voice command..."):
        try:
            files = None

            if audio_bytes:
                # audio_bytes is raw bytes from audio_recorder. Wrap in BytesIO and give a filename.
                buf = io.BytesIO(audio_bytes)
                buf.seek(0)
                # choose a sensible filename and mime
                files = {'audio_file': ('recording.wav', buf, 'audio/wav')}
            elif uploaded_file:
                # read uploaded file bytes
                file_bytes = uploaded_file.read()
                buf = io.BytesIO(file_bytes)
                buf.seek(0)
                mime = uploaded_file.type or 'application/octet-stream'
                files = {'audio_file': (uploaded_file.name, buf, mime)}
            else:
                st.error("No audio file provided")
                return

            # POST to backend
            try:
                resp = requests.post(f"{backend_url.rstrip('/')}/voice-to-dwg", files=files, timeout=120)
            finally:
                # Close bytes buffer to free memory
                try:
                    buf.close()
                except Exception:
                    pass

            if resp.status_code == 200:
                result = resp.json()

                dwg_filename = result.get('dwg_filename') or result.get('dwg_path') or result.get('dwg')

                st.session_state.transcript = result.get('transcript', '')
                st.session_state.parameters = result.get('parameters', {})
                st.session_state.dwg_filename = dwg_filename
                st.session_state.dwg_ready = bool(dwg_filename)

                st.success("✅ Voice command processed successfully!")
            else:
                # show server response text (for debugging)
                st.error(f"❌ Error processing voice command: {resp.status_code} — {resp.text}")

        except requests.Timeout:
            st.error("❌ Request timed out. Try again or increase backend timeout.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def download_dwg_file(dwg_filename, backend_url):
    """Download the generated DWG file from backend cache and present download button."""
    try:
        download_url = f"{backend_url.rstrip('/')}/download-dwg/{dwg_filename}"
        resp = requests.get(download_url, timeout=60)
        if resp.status_code == 200:
            # use the filename returned or a default name
            download_name = f"{dwg_filename}" if dwg_filename else "voice_generated_drawing.dxf"
            st.download_button(
                label="📥 Click to Download DXF",
                data=resp.content,
                file_name=download_name,
                mime="application/dxf",
                use_container_width=True
            )
            st.success("✅ File ready for download!")
        else:
            st.error(f"❌ Error downloading file: {resp.status_code} — {resp.text}")
    except requests.Timeout:
        st.error("❌ Download timed out. Try again.")
    except Exception as e:
        st.error(f"❌ Download error: {str(e)}")

# Demo section
def show_demo():
    st.header("Demo & Examples")
    with st.expander("See Example Parameters"):
        example_params = {
            "room_type": "kitchen",
            "dimensions": {"length": 12, "width": 10, "unit": "feet"},
            "elements": [
                {"type": "door", "position": "east", "size": {"width": 3, "height": 7}},
                {"type": "window", "position": "north", "size": {"width": 4, "height": 3}}
            ],
            "additional_notes": "12x10 kitchen with door on right and window in front"
        }
        st.json(example_params)

if __name__ == "__main__":
    main()
    st.markdown("---")
    show_demo()
