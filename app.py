# app.py — listo para Hugging Face Spaces (Gradio)

import os
import io
import base64
import subprocess
from typing import Tuple

import gradio as gr
from openai import OpenAI
from PIL import Image
from docx import Document


# =========================
# CONFIG OPENAI (HF Secrets)
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "Falta OPENAI_API_KEY. En Hugging Face: Settings → Secrets → OPENAI_API_KEY"
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# Elegí modelos “modernos” y consistentes
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "tts-1")
STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")


# =========================
# FUNCIONES DEL AGENTE
# =========================
def clasificar_texto(texto: str) -> str:
    if not texto or not texto.strip():
        return "Pegá un texto para clasificar."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "Clasificá el texto en: Tecnología, Salud, Educación, Finanzas."},
            {"role": "user", "content": texto},
        ],
    )
    return resp.choices[0].message.content


def texto_a_audio(texto: str) -> str:
    if not texto or not texto.strip():
        return ""

    output_file = "tts.mp3"
    # Streaming a archivo (bien para HF)
    with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice="alloy",
        response_format="mp3",
        input=texto,
    ) as response:
        response.stream_to_file(output_file)

    return output_file


def audio_a_texto(audio_path: str) -> str:
    if not audio_path:
        return "Subí un audio."

    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
        )
    return transcript.text


def traducir_audio(audio_path: str) -> str:
    if not audio_path:
        return "Subí un audio."

    with open(audio_path, "rb") as f:
        transcript = client.audio.translations.create(
            model=STT_MODEL,
            file=f,
        )
    return transcript.text


def analizar_imagen(img: Image.Image) -> str:
    if img is None:
        return "Subí una imagen."

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    resp = client.responses.create(
        model=CHAT_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "¿Qué se observa en la imagen? Describí y, si aplica, inferí contexto."},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
            ],
        }],
    )
    return resp.output_text


def chat_general(mensaje: str) -> str:
    if not mensaje or not mensaje.strip():
        return "Decime algo y charlamos."

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": mensaje}],
    )
    return resp.choices[0].message.content


def mp4_a_texto(mp4_path: str) -> Tuple[str, str, str]:
    """
    - Extrae audio del MP4
    - Segmenta cada 10 minutos
    - Transcribe con Whisper
    - Devuelve: texto, path .txt, path .docx
    """
    if not mp4_path:
        return "Subí un MP4.", "", ""

    os.makedirs("audios", exist_ok=True)

    mp3_path = "audio_principal.mp3"
    txt_path = "transcripcion.txt"
    docx_path = "transcripcion.docx"

    # 1) Extraer audio
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-vn", "-q:a", "2", mp3_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # 2) Segmentar audio (600s = 10 min)
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3_path, "-f", "segment", "-segment_time", "600", "-c", "copy", "audios/parte_%03d.mp3"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    archivos = sorted([f for f in os.listdir("audios") if f.endswith(".mp3")])

    texto_final = []
    for archivo in archivos:
        ruta = os.path.join("audios", archivo)
        with open(ruta, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=audio_file,
            )
        texto_final.append(f"\n--- {archivo} ---\n{transcript.text}")

    texto_unido = "\n".join(texto_final).strip()

    # 3) Guardar TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(texto_unido)

    # 4) Guardar DOCX
    doc = Document()
    for bloque in texto_final:
        doc.add_paragraph(bloque)
    doc.save(docx_path)

    return texto_unido, txt_path, docx_path


# =========================
# UI GRADIO
# =========================
with gr.Blocks(title="Agente Multimodal OpenAI") as demo:
    gr.Markdown("# 🤖 Agente Multimodal con OpenAI (HF Ready)")

    with gr.Tab("📝 Clasificación de Texto"):
        txt_in = gr.Textbox(label="Texto", lines=6)
        out = gr.Textbox(label="Resultado", lines=4)
        gr.Button("Clasificar").click(clasificar_texto, txt_in, out)

    with gr.Tab("🗣️ Texto → Audio"):
        tts_in = gr.Textbox(label="Texto", lines=6)
        tts_out = gr.Audio(label="Audio generado", type="filepath")
        gr.Button("Generar Audio").click(texto_a_audio, tts_in, tts_out)

    with gr.Tab("🎧 Audio → Texto"):
        stt_in = gr.Audio(type="filepath", label="Audio")
        stt_out = gr.Textbox(label="Transcripción", lines=10)
        gr.Button("Transcribir").click(audio_a_texto, stt_in, stt_out)

    with gr.Tab("🌍 Audio → Traducción"):
        tr_in = gr.Audio(type="filepath", label="Audio")
        tr_out = gr.Textbox(label="Traducción", lines=10)
        gr.Button("Traducir").click(traducir_audio, tr_in, tr_out)

    with gr.Tab("🖼️ Análisis de Imagen"):
        img_in = gr.Image(type="pil", label="Imagen")
        img_out = gr.Textbox(label="Análisis", lines=10)
        gr.Button("Analizar").click(analizar_imagen, img_in, img_out)

    with gr.Tab("🎬 MP4 → Transcripción"):
        video_in = gr.Video(label="Subí tu MP4", format="mp4")
        mp4_txt = gr.Textbox(label="Texto transcripto", lines=15)
        mp4_file_txt = gr.File(label="TXT")
        mp4_file_docx = gr.File(label="DOCX")

        gr.Button("Transcribir clase").click(
            mp4_a_texto,
            inputs=video_in,
            outputs=[mp4_txt, mp4_file_txt, mp4_file_docx],
        )

    with gr.Tab("💬 Chat"):
        chat_in = gr.Textbox(label="Mensaje", lines=4)
        chat_out = gr.Textbox(label="Respuesta", lines=10)
        gr.Button("Enviar").click(chat_general, chat_in, chat_out)

# HF Spaces usa PORT automáticamente; Gradio lo toma sin problema.
demo.launch(server_name="0.0.0.0")
