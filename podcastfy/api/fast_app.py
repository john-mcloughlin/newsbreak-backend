"""
FastAPI implementation for Podcastify podcast generation service.
Provides endpoints for podcast generation, audio summary, health checks, and audio file serving.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import shutil
import yaml
from typing import Dict, Any
from pathlib import Path
from ..client import generate_podcast
import uvicorn

# Setup
app = FastAPI()
TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

# Helpers
def load_base_config() -> Dict[Any, Any]:
    config_path = Path(__file__).parent / "podcastfy" / "conversation_config.yaml"
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Warning: Could not load base config: {e}")
        return {}

def merge_configs(base_config: Dict[Any, Any], user_config: Dict[Any, Any]) -> Dict[Any, Any]:
    merged = base_config.copy()
    if 'text_to_speech' in merged and 'text_to_speech' in user_config:
        merged['text_to_speech'].update(user_config.get('text_to_speech', {}))
    for key, value in user_config.items():
        if key != 'text_to_speech' and value is not None:
            merged[key] = value
    return merged

# Endpoint: Podcast Generation
@app.post("/generate")
async def generate_podcast_endpoint(data: dict):
    try:
        os.environ['OPENAI_API_KEY'] = data.get('openai_key', '')
        os.environ['GEMINI_API_KEY'] = data.get('google_key', '')
        os.environ['ELEVENLABS_API_KEY'] = data.get('elevenlabs_key', '')

        base_config = load_base_config()
        tts_model = data.get('tts_model', base_config.get('text_to_speech', {}).get('default_tts_model', 'openai'))
        tts_base_config = base_config.get('text_to_speech', {}).get(tts_model, {})
        voices = data.get('voices', {})
        default_voices = tts_base_config.get('default_voices', {})

        user_config = {
            'creativity': float(data.get('creativity', base_config.get('creativity', 0.7))),
            'conversation_style': data.get('conversation_style', base_config.get('conversation_style', [])),
            'roles_person1': data.get('roles_person1', base_config.get('roles_person1')),
            'roles_person2': data.get('roles_person2', base_config.get('roles_person2')),
            'dialogue_structure': data.get('dialogue_structure', base_config.get('dialogue_structure', [])),
            'podcast_name': data.get('name', base_config.get('podcast_name')),
            'podcast_tagline': data.get('tagline', base_config.get('podcast_tagline')),
            'output_language': data.get('output_language', base_config.get('output_language', 'English')),
            'user_instructions': data.get('user_instructions', base_config.get('user_instructions', '')),
            'engagement_techniques': data.get('engagement_techniques', base_config.get('engagement_techniques', [])),
            'text_to_speech': {
                'default_tts_model': tts_model,
                'model': tts_base_config.get('model'),
                'default_voices': {
                    'question': voices.get('question', default_voices.get('question')),
                    'answer': voices.get('answer', default_voices.get('answer'))
                }
            }
        }

        conversation_config = merge_configs(base_config, user_config)

        result = generate_podcast(
            urls=data.get('urls', []),
            conversation_config=conversation_config,
            tts_model=tts_model,
            longform=bool(data.get('is_long_form', False)),
            read_time=int(data.get('read_time', 3)),
            reading_speed=data.get('reading_speed', 'normal')
        )

        filename = f"podcast_{os.urandom(8).hex()}.mp3"
        output_path = os.path.join(TEMP_DIR, filename)

        if isinstance(result, str) and os.path.isfile(result):
            shutil.copy2(result, output_path)
        elif hasattr(result, 'audio_path') and os.path.isfile(result.audio_path):
            shutil.copy2(result.audio_path, output_path)
        else:
            raise HTTPException(status_code=500, detail="Invalid result format")

        return {"audioUrl": f"/audio/{filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: Serve Audio Files
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    file_path = os.path.join(TEMP_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# Endpoint: Health Check
@app.get("/health")
async def healthcheck():
    return {"status": "healthy"}

# ✅ NEW: /audio-summary endpoint for frontend integration
class SummaryRequest(BaseModel):
    text: str

@app.post("/audio-summary")
async def generate_audio_summary(data: SummaryRequest):
    return {"message": f"Audio summary for: {data.text}"}

# Uvicorn startup
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
