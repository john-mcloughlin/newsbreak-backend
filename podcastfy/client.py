"""
Podcastfy CLI

This module provides a command-line interface for generating podcasts or transcripts
from URLs or existing transcript files. It orchestrates the content extraction,
generation, and text-to-speech conversion processes.
"""

import os
import uuid
import typer
import yaml
from podcastfy.content_parser.content_extractor import ContentExtractor
from podcastfy.content_generator import ContentGenerator
from podcastfy.text_to_speech import TextToSpeech
from podcastfy.utils.config import Config, load_config
from podcastfy.utils.config_conversation import load_conversation_config
from podcastfy.utils.logger import setup_logger
from typing import List, Optional, Dict, Any
import copy

import logging

# Configure logging to show all levels and write to both file and console
""" logging.basicConfig(
    level=logging.DEBUG,  # Show all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('podcastfy.log'),  # Save to file
        logging.StreamHandler()  # Print to console
    ]
) """


logger = setup_logger(__name__)

app = typer.Typer()

os.environ["LANGCHAIN_TRACING_V2"] = "False"

def process_content(
    urls: Optional[List[str]] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    generate_audio: bool = True,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
    text: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key_label: Optional[str] = None,
    topic: Optional[str] = None,
    longform: bool = False,
    read_time: int = 3,  # ✅ NEW
    reading_speed: str = "normal"  # ✅ ADD THIS
):
    """
    Process URLs, a transcript file, image paths, or raw text to generate a podcast or transcript.
    """
    try:
        print("🛠️ [1/6] Loading base configuration (e.g. API keys, output directories)", flush=True)
        if config is None:
            config = load_config()

        print("💬 [2/6] Loading conversation config (e.g. tone, style, voices)", flush=True)
        conv_config = load_conversation_config()

        print("📂 [client.py] 🔧 [2.1] Applying user overrides to conversation config...", flush=True)
        if conversation_config:
            conv_config.configure(conversation_config)
            print(f"📂 [client.py] ✅ Updated conversation config with user values.", flush=True)
        else:
            print("📂 [client.py] ℹ️ No user-provided conversation config to apply.", flush=True)

        print("📂 [client.py] 📁 [2.2] Extracting TTS configuration...", flush=True)
        tts_config = conv_config.get("text_to_speech", {})
        print(f"📂 [client.py] 📄 TTS config: {tts_config}", flush=True)

        print("📂 [client.py] 📂 [2.3] Determining output directories for audio/transcripts...", flush=True)
        output_directories = tts_config.get("output_directories", {})
        print(f"📂 [client.py] 📁 Output directories: {output_directories}", flush=True)

        # 📂 [client.py] 📍 Step 3 — Handle content input: use transcript if available, else prepare generator

        if transcript_file:
            # ✅ If a user-provided transcript is available, use it directly
            logger.info(f"Using transcript file: {transcript_file}")
            print(f"📂 [client.py] 📄 [3.1] Using provided transcript file: {transcript_file}", flush=True)

            with open(transcript_file, "r") as file:
                qa_content = file.read()

            print(f"📂 [client.py] ✅ [3.2] Transcript file loaded successfully. Length: {len(qa_content)} characters",
                  flush=True)

        else:
            # 🧪 No transcript provided — we'll generate one based on URLs, topics, or raw text
            content_extractor = None

            # Determine if extraction is needed (e.g., for short text, URLs, or topic-based content)
            if urls or topic or (text and longform and len(text.strip()) < 100):
                content_extractor = ContentExtractor()
                print("📂 [client.py] 🧠 [3.3] ContentExtractor initialized (for URL/topic/short text expansion)",
                      flush=True)
            else:
                print("📂 [client.py] ℹ️ [3.3] No content extractor needed — using full input text.", flush=True)

            # Initialize the LLM content generator
            content_generator = ContentGenerator(
                is_local=is_local,
                model_name=model_name,
                api_key_label=api_key_label,
                conversation_config=conv_config.to_dict()
            )
            print(f"📂 [client.py] 🔧 [3.4] ContentGenerator initialized with model: {model_name or 'default'}",
                  flush=True)

            combined_content = ""

            if urls:
                print(f"🔍 [3/6] Extracting content from {len(urls)} URL(s)...", flush=True)
                logger.info(f"Processing {len(urls)} links")
                contents = [content_extractor.extract_content(link) for link in urls]
                combined_content += "\n\n".join(contents)
                print("✅ Content extraction complete.", flush=True)

            if text:
                if longform and len(text.strip()) < 100:
                    logger.info("Text too short for direct long-form generation. Extracting context...")
                    print("📚 Expanding short input text into longer form using topic context...", flush=True)
                    expanded_content = content_extractor.generate_topic_content(text)
                    combined_content += f"\n\n{expanded_content}"
                else:
                    combined_content += f"\n\n{text}"

            if topic:
                print(f"🧵 Adding content for topic: {topic}", flush=True)
                topic_content = content_extractor.generate_topic_content(topic)
                combined_content += f"\n\n{topic_content}"

            random_filename = f"transcript_{uuid.uuid4().hex}.txt"
            transcript_filepath = os.path.join(
                output_directories.get("transcripts", "data/transcripts"),
                random_filename,
            )

            # ✅ NEW: Calculate target word count

            words_per_minute = 50 # default

            if read_time:
                words_per_minute = {
                    "slow": 25,
                    "normal": 50,
                    "fast": 75
                }.get(reading_speed, 200)
            target_word_count = read_time * words_per_minute
            print(f"📏 Target word count: ~{target_word_count} words for {read_time} minutes (reading speed = {reading_speed}, words per minute = {words_per_minute})", flush=True)

            print("📝 [4/6] Generating transcript with Q&A model...", flush=True)
            model_used = model_name or conversation_config.get("default_llm_model", "gemini")
            print(f"🧠 Calling LLM API for summarisation: {model_used}", flush=True)
            logger.info(f"Calling LLM API for summarisation: {model_used}")

            qa_content = content_generator.generate_qa_content(
                combined_content,
                image_file_paths=image_paths or [],
                output_filepath=transcript_filepath,
                longform=longform,
                max_words=target_word_count  # ✅ NEW
            )
            print(f"✅ Transcript saved to: {transcript_filepath}", flush=True)

        if generate_audio:
            print("🔊 [5/6] Starting text-to-speech audio generation...", flush=True)

            fallback_order = [tts_model]
            if tts_model != "edge":
                fallback_order.append("edge")
            if tts_model != "elevenlabs":
                fallback_order.append("elevenlabs")

            random_filename = f"podcast_{uuid.uuid4().hex}.mp3"
            audio_file = os.path.join(
                output_directories.get("audio", "data/audio"), random_filename
            )

            for model in fallback_order:
                try:
                    print(f"🔄 Trying TTS model: {model}", flush=True)
                    print(f"🔊 Calling TTS API: {model}", flush=True)
                    logger.info(f"Calling TTS API: {model}")

                    api_key = None
                    if model != "edge":
                        api_key = getattr(config, f"{model.upper().replace('MULTI', '')}_API_KEY")

                    text_to_speech = TextToSpeech(
                        model=model,
                        api_key=api_key,
                        conversation_config=conv_config.to_dict(),
                    )

                    # Check TTS quota up front with a dummy call
                    if not text_to_speech.check_quota():
                        raise RuntimeError("TTS quota is exhausted — skipping this provider before full generation.")

                    text_to_speech.convert_to_speech(qa_content, audio_file)

                    logger.info(f"Podcast generated successfully using {model} TTS model")
                    print(f"✅ [6/6] Podcast saved to: {audio_file}", flush=True)
                    return audio_file  # ✅ Early return if successful

                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "quota" in err_msg.lower():
                        print(f"⚠️ Quota error with {model}. Trying next available TTS...", flush=True)
                        logger.warning(f"TTS quota issue with {model}: {err_msg}")
                        continue  # Try next fallback
                    else:
                        logger.error(f"❌ TTS error with {model}: {err_msg}")
                        raise  # Only raise non-quota errors

            # ❌ If no model succeeded
            raise RuntimeError("❌ Failed to generate audio with all available TTS models.")
        else:
            print(f"✅ [Final] Transcript generation complete. File: {transcript_filepath}", flush=True)
            logger.info(f"Transcript generated successfully: {transcript_filepath}")
            return transcript_filepath

    except Exception as e:
        logger.error(f"An error occurred in the process_content function: {str(e)}")
        print(f"❌ Error in process_content: {str(e)}", flush=True)
        raise

@app.command()
def main(
    urls: list[str] = typer.Option(None, "--url", "-u", help="URLs to process"),
    file: typer.FileText = typer.Option(
        None, "--file", "-f", help="File containing URLs, one per line"
    ),
    read_time: int = typer.Option(
        3, "--read-time", help="Target podcast length in minutes"
    ),
    reading_speed: str = typer.Option("normal", "--reading-speed", help="Reading speed: slow, normal, fast"),
    transcript: typer.FileText = typer.Option(
        None, "--transcript", "-t", help="Path to a transcript file"
    ),
    tts_model: str = typer.Option(
        None, "--tts-model", "-tts", help="TTS model to use (openai, elevenlabs, edge, or gemini)"
    ),
    transcript_only: bool = typer.Option(
        False, "--transcript-only", help="Generate only a transcript without audio"
    ),
    conversation_config_path: str = typer.Option(
        None, "--conversation-config", "-cc", help="Path to custom conversation configuration YAML file"
    ),
    image_paths: List[str] = typer.Option(
        None, "--image", "-i", help="Paths to image files to process"
    ),
    is_local: bool = typer.Option(
        False, "--local", "-l", help="Use a local LLM instead of a remote one (http://localhost:8080)"
    ),
    text: str = typer.Option(
        None, "--text", "-txt", help="Raw text input to be processed"
    ),
    llm_model_name: str = typer.Option(
        None, "--llm-model-name", "-m", help="LLM model name for transcript generation"
    ),
    api_key_label: str = typer.Option(
        None, "--api-key-label", "-k", help="Environment variable name for LLMAPI key"
    ),
    topic: str = typer.Option(
        None, "--topic", "-tp", help="Topic to generate podcast about"
    ),
    longform: bool = typer.Option(
        False, "--longform", "-lf", help="Generate long-form content (only available for text input without images)"
    ),
):

    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, image files, or raw text.
    """
    try:
        config = load_config()
        main_config = config.get("main", {})

        conversation_config = None
        # Load conversation config if provided
        if conversation_config_path:
            with open(conversation_config_path, "r") as f:
                conversation_config: Dict[str, Any] | None = yaml.safe_load(f)

        # Use default TTS model from conversation config if not specified
        if tts_model is None:
            tts_config = load_conversation_config().get("text_to_speech", {})
            tts_model = tts_config.get("default_tts_model", "gemini")

        if transcript:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            final_output = process_content(
                transcript_file=transcript.name,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                conversation_config=conversation_config,
                config=config,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform
            )
        else:
            urls_list = urls or []
            if file:
                urls_list.extend([line.strip() for line in file if line.strip()])

            if not urls_list and not image_paths and not text and not topic:
                raise typer.BadParameter(
                    "No input provided. Use --url, --file, --transcript, --image, --text, or --topic."
                )

            final_output = process_content(
                urls=urls_list,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=config,
                conversation_config=conversation_config,
                image_paths=image_paths,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform
            )

        if transcript_only:
            typer.echo(f"Transcript generated successfully: {final_output}")
        else:
            typer.echo(
                f"Podcast generated successfully using {tts_model} TTS model: {final_output}"
            )

    except Exception as e:
        typer.echo(f"An error occurred: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()


def generate_podcast(
    urls: Optional[List[str]] = None,
    url_file: Optional[str] = None,
    transcript_file: Optional[str] = None,
    tts_model: Optional[str] = None,
    transcript_only: bool = False,
    config: Optional[Dict[str, Any]] = None,
    conversation_config: Optional[Dict[str, Any]] = None,
    image_paths: Optional[List[str]] = None,
    is_local: bool = False,
    text: Optional[str] = None,
    llm_model_name: Optional[str] = None,
    api_key_label: Optional[str] = None,
    topic: Optional[str] = None,
    longform: bool = False,
    read_time: int = 3,
    reading_speed: str = "normal",
) -> Optional[str]:
    """
    Generate a podcast or transcript from a list of URLs, a file containing URLs, a transcript file, or image files.

    Args:
        urls (Optional[List[str]]): List of URLs to process.
        url_file (Optional[str]): Path to a file containing URLs, one per line.
        transcript_file (Optional[str]): Path to a transcript file.
        tts_model (Optional[str]): TTS model to use ('gemini' [default], 'elevenlabs', 'edge', or 'openai').
        transcript_only (bool): Generate only a transcript without audio. Defaults to False.
        config (Optional[Dict[str, Any]]): User-provided configuration dictionary.
        conversation_config (Optional[Dict[str, Any]]): User-provided conversation configuration dictionary.
        image_paths (Optional[List[str]]): List of image file paths to process.
        is_local (bool): Whether to use a local LLM. Defaults to False.
        text (Optional[str]): Raw text input to be processed.
        llm_model_name (Optional[str]): LLM model name for content generation.
        api_key_label (Optional[str]): Environment variable name for LLM API key.
        topic (Optional[str]): Topic to generate podcast about.

    Returns:
        Optional[str]: Path to the final podcast audio file, or None if only generating a transcript.
    """
    try:
        print("Generating podcast...")
        # Load default config
        default_config = load_config()

        # Update config if provided
        if config:
            if isinstance(config, dict):
                # Create a deep copy of the default config
                updated_config = copy.deepcopy(default_config)
                # Update the copy with user-provided values
                updated_config.configure(**config)
                default_config = updated_config
            elif isinstance(config, Config):
                # If it's already a Config object, use it directly
                default_config = config
            else:
                raise ValueError(
                    "Config must be either a dictionary or a Config object"
                )

        if not conversation_config:
            conversation_config = load_conversation_config().to_dict()

        main_config = default_config.config.get("main", {})

        # Use provided tts_model if specified, otherwise use the one from config
        if tts_model is None:
            tts_model = conversation_config.get("default_tts_model", "gemini")

        if transcript_file:
            if image_paths:
                logger.warning("Image paths are ignored when using a transcript file.")
            return process_content(
                transcript_file=transcript_file,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=default_config,
                conversation_config=conversation_config,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform,
                read_time = read_time,
                reading_speed = reading_speed
            )
        else:
            urls_list = urls or []
            if url_file:
                with open(url_file, "r") as file:
                    urls_list.extend([line.strip() for line in file if line.strip()])

            if not urls_list and not image_paths and not text and not topic:
                raise ValueError(
                    "No input provided. Please provide either 'urls', 'url_file', "
                    "'transcript_file', 'image_paths', 'text', or 'topic'."
                )

            return process_content(
                urls=urls_list,
                tts_model=tts_model,
                generate_audio=not transcript_only,
                config=default_config,
                conversation_config=conversation_config,
                image_paths=image_paths,
                is_local=is_local,
                text=text,
                model_name=llm_model_name,
                api_key_label=api_key_label,
                topic=topic,
                longform=longform,
                read_time=read_time,
                reading_speed=reading_speed
            )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
