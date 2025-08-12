import streamlit as st
import requests
import tempfile
import os
import time
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
            if st.button("💾 Download DXF File", type="secondary", use_container_width=True):
                download_dwg_file(st.session_state.dwg_path, BACKEND_URL)

def process_voice_command(audio_bytes, uploaded_file, backend_url):
    """Process the voice command through the complete pipeline"""
    
    with st.spinner("Processing your voice command..."):
        temp_path = None
        file_handle = None
        
        try:
            # Prepare file for upload
            if audio_bytes:
                # Save recorded audio
                temp_path = tempfile.mktemp(suffix='.wav')
                with open(temp_path, 'wb') as f:
                    f.write(audio_bytes)
                
                # Open file for upload
                file_handle = open(temp_path, 'rb')
                files = {'audio_file': ('recording.wav', file_handle, 'audio/wav')}
            elif uploaded_file:
                files = {'audio_file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
            else:
                st.error("No audio file provided")
                return
            
            # Call the complete pipeline endpoint
            response = requests.post(f"{backend_url}/voice-to-dwg", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                # Store in session state
                st.session_state.transcript = result['transcript']
                st.session_state.parameters = result['parameters']
                st.session_state.dwg_path = result['dwg_path']
                st.session_state.dwg_ready = True
                
                st.success("✅ Voice command processed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error processing voice command: {response.text}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        finally:
            # Clean up - close file handle first, then delete file
            if file_handle:
                try:
                    file_handle.close()
                except:
                    pass
            
            if temp_path and os.path.exists(temp_path):
                try:
                    # Add small delay
                    time.sleep(0.1)
                    os.remove(temp_path)
                except PermissionError:
                    # If still locked, try again after longer delay
                    try:
                        time.sleep(0.5)
                        os.remove(temp_path)
                    except:
                        # If still can't delete, log but don't crash
                        st.warning(f"Could not clean up temporary file: {temp_path}")
                except Exception:
                    pass

def download_dwg_file(dwg_path, backend_url):
    """Download the generated DWG file"""
    try:
        download_url = f"{backend_url}/download-dwg/{dwg_path}"
        response = requests.get(download_url)
        
        if response.status_code == 200:
            st.download_button(
                label="📥 Click to Download DXF",
                data=response.content,
                file_name="voice_generated_drawing.dxf",
                mime="application/octet-stream",
                use_container_width=True
            )
            st.success("✅ File ready for download!")
        else:
            st.error("❌ Error downloading file")
    
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