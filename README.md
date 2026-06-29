<h1 align="center">
  <img src="https://github.com/CYFARE/PDXTRACT/blob/main/assets/PDXTRACT.png" alt="PDXTRACT Logo">
</h1>

<h2 align="center">
  <img src="https://img.shields.io/badge/-GPLv2.0-61DAFB?style=for-the-badge" alt="License: GPLv2.0">&nbsp;
</h2>

**PDXTRACT** is a professional, modular PDF extractor built around the proven **OCR + deterministic regex** strategy. It is designed for exhaustive accuracy on dense documents and supports multiple local inference backends.

## Why OCR + Regex?

Vision-Language Models (VLMs) can skip, hallucinate, or stop early on cluttered pages. OCR models like **GOT-OCR 2.0** transcribe every character mindlessly and deterministically. Feeding that raw text into regex guarantees that no email, phone, or URL is missed.

## Features

- **Multiple OCR backends**
  - **Native GOT-OCR 2.0** via Hugging Face Transformers (fastest, optional)
  - **Ollama** (any vision/OCR model: `llama3.2-vision`, custom GGUFs, etc.)
  - **llama.cpp server** (OpenAI-compatible API for GLM-OCR, Deepseek-OCR, HunyuanOCR, Gemma-3/4 vision, etc.)
- **Extraction types**: email, phone, URL, and custom regex patterns
- **Strategies**: `ocr_regex` (default), `hybrid` (embedded text shortcut), `vlm_regex`
- **Resume support**: session log + incremental JSONL temp file
- **Rich CLI**: progress bars, model listing, multiple output formats
- **Outputs**: JSON, CSV, and flat TXT

## Quick Start

```bash
git clone https://github.com/CYFARE/PDXTRACT.git
cd PDXTRACT
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run with Ollama

```bash
ollama pull llama3.2-vision
ollama serve
python xtract.py
```

### Run with native GOT-OCR 2.0 (optional, best speed/accuracy)

```bash
python -m pip install -r requirements-got.txt
# edit config.json: provider = "got_ocr20_native", model = "stepfun-ai/GOT-OCR2_0"
python xtract.py
```

> Native GOT-OCR 2.0 automatically falls back to eager attention on systems without `flash-attn`.

### Run with llama.cpp server

```bash
llama-server -hf ggml-org/GLM-OCR-GGUF
# edit config.json: provider = "llama_cpp", model = "ggml-org/GLM-OCR-GGUF"
python xtract.py
```

## Usage

```bash
# Process PDFs (uses config.json)
python xtract.py

# Override provider/model on the fly
python xtract.py process --provider ollama --model llama3.2-vision --max-workers 2

# List available models
python xtract.py list-models

# Help
python xtract.py --help
python xtract.py process --help
```

## Configuration

Edit `config.json`:

```json
{
  "provider": "ollama",
  "model": "llama3.2-vision",
  "strategy": "ocr_regex",
  "extractors": ["email"],
  "custom_regex": {},
  "prompt": "Transcribe all text visible in this image. Output only the raw text, no explanations.",
  "ollama_url": "http://127.0.0.1:11434",
  "llama_cpp_url": "http://127.0.0.1:8080",
  "input_folder": "./pdfs",
  "output_file": "output/extracted_data.json",
  "output_csv": "output/extracted_data.csv",
  "output_txt": "output/emails.txt",
  "session_file": "aiocr_session.log",
  "max_workers": 4,
  "page_dpi": 200,
  "use_embedded_text": true,
  "include_ocr_text": false
}
```

### Custom regex

```json
{
  "extractors": ["email", "custom_case_number"],
  "custom_regex": {
    "custom_case_number": "Case\\s+#?\\s*(\\d{4}-\\d{4})"
  }
}
```

## Strategies

| Strategy | Behavior |
|---|---|
| `ocr_regex` | Always render + OCR, then regex. Most reliable for scanned documents. |
| `hybrid` | Extract embedded text first; OCR only if no matches. Fastest for text-based PDFs. |
| `vlm_regex` | Send image to a VLM with the configured prompt, then regex. Compatibility mode. |

## Project Layout

```
pdxtract/
  providers/      # OCR backend implementations
  core/           # rendering, extraction, pipeline
  output/         # JSON/JSONL/CSV/session writers
  utils/          # logging
  cli.py          # command-line interface
config.json       # user configuration
xtract.py         # backward-compatible wrapper
```

## Migrating from PDXTRACT v1

- `xtract.py` still works as a wrapper and now delegates to the modular package.
- `config.json` keys remain backward-compatible, but `prompt` now means **transcription prompt** (e.g., "Transcribe all text..."), not "Extract all emails...". The actual extraction is done by deterministic regex.
- The default output file changed to `output/extracted_data.json`.
- Session log default remains `aiocr_session.log` so existing resume state is preserved.

## License

GPLv2.0

## Support

Boost Cyfare by spreading the word: https://cyfare.net/apps/Social/
