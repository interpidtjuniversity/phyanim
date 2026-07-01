import os
import hashlib
import base64
import requests
from manim_voiceover.services.base import SpeechService


class MinimaxSpeechService(SpeechService):
    """
    MiniMax TTS 语音服务（基于官方 v1/t2a_v2 接口）
    使用 WAV 格式，通过文件头魔数验证，不依赖 mutagen 解析。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "speech-2.8-hd",
        voice_id: str = "male-qn-qingse",
        speed: float = 1.0,
        vol: float = 1.0,
        pitch: float = 0.0,
        emotion: str = None,
        sample_rate: int = 32000,
        bitrate: int = 128000,
        audio_format: str = "wav",
        channel: int = 1,
        pronunciation_dict: dict = None,
        subtitle_enable: bool = False,
        base_url: str = "https://api.minimaxi.com",
    ):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.voice_id = voice_id
        self.speed = speed
        self.vol = vol
        self.pitch = pitch
        self.emotion = emotion
        self.sample_rate = sample_rate
        self.bitrate = bitrate
        self.audio_format = audio_format
        self.channel = channel
        self.pronunciation_dict = pronunciation_dict
        self.subtitle_enable = subtitle_enable
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/v1/t2a_v2"
        self.ext_map = {"mp3": ".mp3", "wav": ".wav", "flac": ".flac"}

        # 确保 cache_dir 是绝对路径
        if self.cache_dir is None:
            self.cache_dir = "media/voiceovers/"
        self.cache_dir = os.path.abspath(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def generate_from_text(self, text: str, cache_dir: str = None, **kwargs):
        if cache_dir is not None:
            self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

        ext = self.ext_map.get(self.audio_format, ".wav")
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        filename = f"{text_hash}{ext}"                     # 纯文件名
        audio_path = os.path.abspath(os.path.join(self.cache_dir, filename))

        # 检查缓存文件是否存在且看起来有效（检查魔数）
        if os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                header = f.read(12)  # WAV 头前12字节应为 RIFF...WAVE
                if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                    return {
                        "original_audio": filename,
                        "final_audio": filename,
                        "sample_rate": self.sample_rate,
                    }
                else:
                    try:
                        os.remove(audio_path)  # 无效则删除
                    except PermissionError:
                        pass  # Windows 文件锁定时忽略，后续会覆盖

        # --- 调用 MiniMax API ---
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        voice_setting = {
            "voice_id": self.voice_id,
            "speed": float(self.speed),
            "vol": float(self.vol),
            "pitch": int(self.pitch),
        }
        if self.emotion is not None:
            voice_setting["emotion"] = self.emotion

        audio_setting = {
            "sample_rate": self.sample_rate,
            "bitrate": self.bitrate,
            "format": self.audio_format,
            "channel": self.channel,
        }

        payload = {
            "model": self.model,
            "text": text,
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            "subtitle_enable": self.subtitle_enable,
        }
        if self.pronunciation_dict is not None:
            payload["pronunciation_dict"] = self.pronunciation_dict

        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"MiniMax API 请求失败: {e}") from e

        result = response.json()
        base_resp = result.get("base_resp", {})
        status_code = base_resp.get("status_code")
        if status_code != 0:
            status_msg = base_resp.get("status_msg", "未知错误")
            raise RuntimeError(f"MiniMax API 返回错误 (code={status_code}): {status_msg}")

        audio_hex = result.get("data", {}).get("audio")
        if not audio_hex:
            raise RuntimeError("MiniMax API 返回的音频数据为空")

        audio_bytes = bytes.fromhex(audio_hex)

        # 写入文件
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # 简单验证：检查魔数
        with open(audio_path, "rb") as f:
            header = f.read(12)
            if not (header[:4] == b"RIFF" and header[8:12] == b"WAVE"):
                try:
                    os.remove(audio_path)
                except PermissionError:
                    pass  # Windows 文件锁定时忽略
                raise RuntimeError("下载的文件不是有效的 WAV 格式")

        extra_info = result.get("extra_info", {})
        sample_rate_from_api = extra_info.get("audio_sample_rate")
        final_sample_rate = int(sample_rate_from_api) if sample_rate_from_api is not None else self.sample_rate

        return {
            "original_audio": audio_path,
            "final_audio": filename,
            "sample_rate": final_sample_rate,
        }