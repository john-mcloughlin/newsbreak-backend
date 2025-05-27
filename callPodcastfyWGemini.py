
import sys
import os
import shutil
import typer
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

# Load environment variables first
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def check_api_keys():
    """Check if required API keys are available"""
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    elevenlabs_key = os.getenv('ELEVENLABS_API_KEY')
    
    if not gemini_key:
        log("❌ Missing GEMINI_API_KEY - this is required for Podcastify")
        return False
    
    if not openai_key:
        log("❌ Missing OPENAI_API_KEY")
        return False
        
    if not elevenlabs_key:
        log("❌ Missing ELEVENLABS_API_KEY")
        return False
    
    log("✅ All required API keys found")
    return True

def setup_podcastfy_path():
    """Setup and validate the Podcastfy path"""
    try:
        # Try to find podcastfy in the current directory
        current_dir = os.path.dirname(__file__)
        podcastfy_path = os.path.join(current_dir, "podcastfy")
        
        if os.path.isdir(podcastfy_path):
            log(f"✅ Found podcastfy directory: {podcastfy_path}")
            return podcastfy_path
        
        log(f"❌ Podcastfy directory not found at: {podcastfy_path}")
        return None
        
    except Exception as e:
        log(f"❌ Error setting up podcastfy path: {str(e)}")
        return None

def import_podcastfy():
    """Import podcastfy with proper error handling"""
    try:
        podcastfy_path = setup_podcastfy_path()
        if not podcastfy_path:
            log("❌ Could not setup podcastfy path")
            return None
        
        if podcastfy_path not in sys.path:
            sys.path.insert(0, podcastfy_path)
            log(f"📦 Added to Python path: {podcastfy_path}")
        
        from podcastfy.client import generate_podcast
        log("✅ Successfully imported podcastfy.client")
        return generate_podcast
        
    except ImportError as e:
        log(f"❌ Failed to import podcastfy: {str(e)}")
        log("💡 Make sure podcastfy is properly installed")
        return None
    except Exception as e:
        log(f"❌ Unexpected error importing podcastfy: {str(e)}")
        return None

def generate_combined_podcast(urls, title, read_time=3, reading_speed="normal"):
    """Generate podcast with better error handling"""
    try:
        log(f"📡 Starting podcast generation for {len(urls)} URLs...")
        
        # Check API keys first
        if not check_api_keys():
            log("❌ Missing required API keys")
            return None
        
        # Import podcastfy
        generate_podcast = import_podcastfy()
        if not generate_podcast:
            log("❌ Could not import podcastfy")
            return None
        
        log("🧠 Invoking Gemini model for summarisation and synthesis...")

        # Generate the podcast
        audio_file = generate_podcast(
            urls=urls,
            longform=True,
            read_time=read_time,
            reading_speed=reading_speed
        )

        if not audio_file:
            log("❌ No audio file returned from generate_podcast")
            return None
        
        if not os.path.exists(audio_file):
            log(f"❌ Generated audio file does not exist: {audio_file}")
            return None

        log("🎙️ TTS + stitching complete.")
        log("📦 Copying final podcast file...")

        # Create output directory
        output_dir = os.path.join("data", "audio")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{title.replace(' ', '_')}.mp3")
        
        # Copy the file
        shutil.copy2(audio_file, output_path)
        
        # Verify the copy
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            log(f"✅ Podcast saved to: {output_path} (size: {file_size} bytes)")
            return output_path
        else:
            log(f"❌ Failed to copy file to: {output_path}")
            return None
            
    except Exception as e:
        log(f"❌ Error generating podcast: {str(e)}")
        import traceback
        log(f"❌ Traceback: {traceback.format_exc()}")
        return None

app = typer.Typer()

@app.command()
def main(
    urls: List[str] = typer.Argument(..., help="List of article URLs"),
    read_time: int = typer.Option(3, "--read-time", help="Podcast target length in minutes"),
    reading_speed: str = typer.Option("normal", "--reading-speed", help="Reading speed")
):
    """Main function to generate podcast from URLs"""
    podcast_title = "Combined_News_Podcast"

    log("🚀 Script started.")
    log(f"🔎 Preparing to process {len(urls)} article(s)...")
    
    for i, url in enumerate(urls, start=1):
        log(f"   [{i}] {url}")

    # Check environment
    log("🔍 Checking environment...")
    log(f"Python version: {sys.version}")
    log(f"Working directory: {os.getcwd()}")
    log(f"Script location: {__file__}")

    # Generate podcast
    path = generate_combined_podcast(urls, podcast_title, read_time, reading_speed)

    if path:
        log(f"✅ Success! Final path: {path}")
        print(path, flush=True)  # This is what the Node.js script looks for
        sys.exit(0)
    else:
        log("❌ Error: No path returned.")
        print("❌ Error: Podcast generation failed", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    try:
        app()
    except Exception as e:
        log(f"❌ Fatal error: {str(e)}")
        import traceback
        log(f"❌ Traceback: {traceback.format_exc()}")
        sys.exit(1)
