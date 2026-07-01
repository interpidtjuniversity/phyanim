from phyanim.server import ServerConfig, LLMProvider, create_app
from phyanim.voiceover import TTSConfig

config = ServerConfig(
    llm_provider=LLMProvider.DEEPSEEK,
    llm_api_key="sk-77e1188b130f4187b2c66ec583a75616",
    tts=TTSConfig(provider="minimax", api_key="sk-api-3Zu_GYQdFjIoXCXFHGLwcjXL7sUOQGSiAAmbTns5cLpa36Xk9-F1fdezpn9hAwWEDFiLFlsMnavkbh-_GL5oJHOd8zhmtbIgm7dcNhi2keybZ5OtJ0XAOF4", voice_id="male-qn-qingse"),
    code_output_dir="E:\\phyanim\\outputs\\code",
    video_output_dir="E:\\phyanim\\outputs\\video"
)
app = create_app(config)
app.run(host="0.0.0.0", port=5000)