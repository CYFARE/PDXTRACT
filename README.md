<h1 align="center">
  <img src="https://github.com/CYFARE/PDXTRACT/blob/main/assets/PDXTRACT.png" alt="PDXTRACT Logo">
</h1>

<h2 align="center">
  <img src="https://img.shields.io/badge/-GPLv2.0-61DAFB?style=for-the-badge" alt="License: GPLv2.0">&nbsp;
</h2>

**PDXTRACT** is a simple script to extract information from pdf files using prompts, ollama local api server and llama vision model that fits in 8GB GPU VRAM.

**Intention To Make This**: Initially grabbed 100GB worth of US Govt. document corpus and wanted to extract all emails, names and locations in all pdf files. Then parse it though mx records and mail bounce back to verify validity and make huge list of valid emails. lol.

## Setup & Usage

### Setup

```bash
cd ~ && git clone https://github.com/CYFARE/PDXTRACT.git
cd PDXTRACT
python3 -m venv venv # Or use 'python' depending on your system
source venv/bin/activate # On Windows use `venv\Scripts\activate`
# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Usage

- Install Ollama locally (no matter what way you install, make sure the api server url is accessible)
- Run in terminal: ```ollama pull llama3.2-vision```
- Run in terminal: ```ollama serve```
- Verify Ollama API server to be running (default url: http://127.0.0.1:11434/).
- Make changes to config.json. Default values mentioned below with a sample query that properly extracts all emails in a PDF file.:

```bash
{
  "provider": "ollama",
  "model": "llama3.2-vision",
  "ollama_url": "http://127.0.0.1:11434",
  "prompt": "Extract all emails found on the page. List each email address on a new line and only provide the email address without any text or explanations. Only emails. If there are no emails found, only output 'NO EMAIL' exactly.",
  "input_folder": "./pdfs",
  "output_file": "output/ollama_extracted_data.json",
  "max_workers": 4
}
```

- Just run:

```bash
python xtract.py
```

- If you are automating using this script and want to change config values on the fly without touching config.json, use arguments:

```bash
python xtract.py --help
```

## Support

Boost Cyfare by spreading a word and considering your support: https://cyfare.net/apps/Social/

## License

GPLv2.0
