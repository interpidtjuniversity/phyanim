"""PhyAnim server layer.

Provides a Flask application that exposes the /generate_video endpoint,
centralizing LLM provider selection and TTS configuration.

Usage::

    from phyanim.server import ServerConfig, LLMProvider, create_app
    from phyanim.voiceover import TTSConfig

    config = ServerConfig(
        llm_provider=LLMProvider.DEEPSEEK,
        llm_api_key="sk-...",
        tts=TTSConfig(provider="minimax", api_key="...", voice_id="male-qn-qingse"),
    )
    app = create_app(config)
    app.run(host="0.0.0.0", port=5000)
"""

from phyanim.server.config import ServerConfig, LLMProvider
from phyanim.server.app import create_app

__all__ = ["ServerConfig", "LLMProvider", "create_app"]
