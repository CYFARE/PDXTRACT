import argparse
import base64
import io
import json
import logging
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import ollama
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3
CONFIG_FILE = "config.json"
SESSION_FILE = "aiocr_session.log" # Stores names of processed files
TEMP_OUTPUT_SUFFIX = "_temp.jsonl" # Suffix for intermediate JSON Lines file

# Simple regex for basic email format validation (adjust as needed)
EMAIL_REGEX = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
NEGATIVE_RESPONSES = {"NO EMAIL", "NO EMAILS", "NO EMAIL ADDRESS", "NO EMAIL ADDRESSES FOUND", "NO EMAILS ARE PRESENT", "NO EMAILS WERE FOUND"}


def get_image_bytes_from_page(page):
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    return img_bytes

def process_page_with_ollama(client, model, prompt, image_bytes, page_num, pdf_path):
    attempt = 0
    pdf_filename = os.path.basename(pdf_path)
    while attempt < MAX_RETRIES:
        try:
            response = client.chat(
                model=model,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_bytes]
                    },
                ]
            )
            extracted_text = response['message']['content']
            logging.info(f"Successfully processed Page {page_num + 1} of {pdf_filename} using Ollama model {model}")
            return extracted_text
        except ollama.ResponseError as e:
            logging.warning(f"Ollama API error on page {page_num + 1} of {pdf_filename} (Attempt {attempt + 1}/{MAX_RETRIES}): {e.error}")
            if "model" in str(e.error).lower() and "not found" in str(e.error).lower():
                 logging.error(f"Model '{model}' not found locally. Pull it with 'ollama pull {model}'. Stopping retries for this page.")
                 return None # No point retrying if model is missing
            attempt += 1
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(f"Failed page {page_num + 1} of {pdf_filename} after retries due to Ollama API error: {e.error}")
                return None
        except Exception as e:
            logging.warning(f"Unexpected error processing page {page_num + 1} of {pdf_filename} with Ollama (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            attempt += 1
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                 logging.error(f"Failed page {page_num + 1} of {pdf_filename} after retries due to unexpected error: {e}")
                 return None
    return None


def process_pdf(pdf_path, ollama_client, model, prompt):
    """Processes a single PDF file using Ollama, extracting info page by page."""
    pdf_filename = os.path.basename(pdf_path)
    logging.info(f"Starting processing for PDF: {pdf_filename}")
    pdf_results = {"pdf_filename": pdf_filename, "pages": []}
    doc = None
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        logging.info(f"Found {num_pages} pages in {pdf_filename}.")

        for i in range(num_pages):
            page_start_time = time.time()
            logging.debug(f"Processing Page {i + 1}/{num_pages} of {pdf_filename}...")
            page = doc.load_page(i)
            page_result_data = {"page": i + 1, "raw_extracted_info": None, "status": "failed"} # Changed key name
            try:
                image_bytes = get_image_bytes_from_page(page)
                if not image_bytes:
                     raise ValueError("Extracted image bytes are empty.")

                page_result = process_page_with_ollama(ollama_client, model, prompt, image_bytes, i, pdf_path)

                if page_result:
                    page_result_data["raw_extracted_info"] = page_result # Store raw output
                    page_result_data["status"] = "success"
                else:
                    page_result_data["raw_extracted_info"] = f"Error: Failed to process page with Ollama after {MAX_RETRIES} attempts."

            except Exception as page_err:
                logging.error(f"Error preparing or processing page {i + 1} of {pdf_filename}: {page_err}")
                page_result_data["raw_extracted_info"] = f"Error: Failed processing page {i+1}. Details: {page_err}"

            pdf_results["pages"].append(page_result_data)
            page_end_time = time.time()
            logging.debug(f"Finished page {i+1} of {pdf_filename} in {page_end_time - page_start_time:.2f} seconds.")

        logging.info(f"Finished PDF processing stage for: {pdf_filename}")
        return pdf_filename, pdf_results

    except Exception as e:
        logging.error(f"Failed to open or process PDF {pdf_filename}: {e}")
        return pdf_filename, {"pdf_filename": pdf_filename, "status": "failed", "error": str(e), "pages": []}
    finally:
        if doc:
            doc.close()


def load_config():
    """Loads configuration from config.json."""
    config = {}
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        logging.info(f"Loaded configuration from {CONFIG_FILE}")
    except FileNotFoundError:
        logging.warning(f"{CONFIG_FILE} not found. Using command-line arguments and defaults where possible.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding {CONFIG_FILE}. Please ensure it's valid JSON. Proceeding without config file.")
    except Exception as e:
        logging.error(f"Error reading {CONFIG_FILE}: {e}. Proceeding without config file.")
    return config

def load_processed_files():
    """Loads the set of already processed PDF filenames from the session file."""
    processed = set()
    try:
        with open(SESSION_FILE, 'r') as f:
            for line in f:
                processed.add(line.strip())
        logging.info(f"Loaded {len(processed)} processed files from {SESSION_FILE}")
    except FileNotFoundError:
        logging.info(f"{SESSION_FILE} not found. Starting fresh session.")
    return processed

def save_processed_file(filename):
    """Appends a successfully processed filename to the session file."""
    try:
        with open(SESSION_FILE, 'a') as f:
            f.write(filename + '\n')
    except IOError as e:
        logging.error(f"Failed to write {filename} to session file {SESSION_FILE}: {e}")

def append_result_to_jsonl(result_data, temp_output_file):
    """Appends a single PDF result object to the JSON Lines temp file."""
    try:
        with open(temp_output_file, 'a') as f:
            json.dump(result_data, f)
            f.write('\n')
        return True
    except IOError as e:
        logging.error(f"Failed to append result for {result_data.get('pdf_filename', 'N/A')} to {temp_output_file}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error appending result for {result_data.get('pdf_filename', 'N/A')} to {temp_output_file}: {e}")
        return False

def clean_jsonl_output(temp_output_file, final_output_file):
    """Reads the JSONL temp file, cleans data, and writes final JSON."""
    logging.info(f"Starting final cleaning of data from {temp_output_file}")
    cleaned_results = []
    try:
        with open(temp_output_file, 'r') as f_in:
            for line in f_in:
                try:
                    pdf_result = json.loads(line.strip())
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON line in {temp_output_file}: {line.strip()}")
                    continue

                if pdf_result.get("status") == "failed" or not pdf_result.get("pages"):
                    logging.warning(f"Skipping failed or empty PDF result: {pdf_result.get('pdf_filename')}")
                    continue

                cleaned_pages = []
                for page in pdf_result.get("pages", []):
                    if page.get("status") != "success" or not page.get("raw_extracted_info"):
                        continue

                    raw_info = page["raw_extracted_info"].strip()
                    # Check against negative responses (case-insensitive)
                    if raw_info.upper() in NEGATIVE_RESPONSES:
                        continue # Skip page if Ollama said no emails

                    potential_emails = []
                    # Split potentially multi-line emails and clean them up
                    for email_candidate in raw_info.splitlines():
                        email_candidate = email_candidate.strip()
                        # Basic regex check and ignore known negative responses again just in case
                        if re.match(EMAIL_REGEX, email_candidate) and email_candidate.upper() not in NEGATIVE_RESPONSES:
                             potential_emails.append(email_candidate)

                    if potential_emails:
                         cleaned_pages.append({
                             "page": page["page"],
                             "extracted_emails": potential_emails # Store as a list
                         })

                # Only keep the PDF result if it has pages with cleaned emails
                if cleaned_pages:
                    cleaned_results.append({
                        "pdf_filename": pdf_result["pdf_filename"],
                        "pages": cleaned_pages
                    })

    except FileNotFoundError:
        logging.error(f"Temporary output file {temp_output_file} not found during cleaning.")
        return
    except Exception as e:
        logging.error(f"Error during cleaning process reading {temp_output_file}: {e}")
        # Proceed to save whatever was cleaned successfully so far, or handle error appropriately
        pass # Allow saving partially cleaned data below if desired

    # Save the final cleaned data
    try:
        output_dir = os.path.dirname(os.path.abspath(final_output_file))
        if output_dir and not os.path.exists(output_dir):
             os.makedirs(output_dir)

        with open(final_output_file, 'w') as f_out:
            json.dump(cleaned_results, f_out, indent=4)
        logging.info(f"Successfully cleaned data and saved to {final_output_file}")

        # Optionally remove the temporary file after successful cleaning and saving
        try:
            os.remove(temp_output_file)
            logging.info(f"Removed temporary file: {temp_output_file}")
        except OSError as e:
            logging.warning(f"Could not remove temporary file {temp_output_file}: {e}")

    except IOError as e:
        logging.error(f"Failed to write final cleaned results to {final_output_file}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while saving cleaned results: {e}")


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Perform multi-page OCR and extraction on PDFs using Ollama with resume support.")
    parser.add_argument("--provider", default=config.get("provider", "ollama"), help="AI provider ('ollama').", choices=['ollama'])
    parser.add_argument("--model", default=config.get("model"), help="Ollama model name. Overrides config.")
    parser.add_argument("--ollama_url", default=config.get("ollama_url"), help="URL for Ollama server. Overrides config.")
    parser.add_argument("--prompt", default=config.get("prompt"), help="Extraction prompt. Overrides config.")
    parser.add_argument("--input_folder", default=config.get("input_folder"), help="Input PDF folder. Overrides config.")
    parser.add_argument("--output_file", default=config.get("output_file", "ollama_extracted_data.json"), help="Final cleaned JSON output file path.")
    parser.add_argument("--max_workers", type=int, default=config.get("max_workers", 2), help="Max concurrent workers.")

    args = parser.parse_args()

    final_config = config.copy()
    if args.model: final_config['model'] = args.model
    if args.ollama_url: final_config['ollama_url'] = args.ollama_url
    if args.prompt: final_config['prompt'] = args.prompt
    if args.input_folder: final_config['input_folder'] = args.input_folder
    final_config['output_file'] = args.output_file
    final_config['max_workers'] = args.max_workers
    final_config['provider'] = args.provider # Keep provider logic

    required_keys = ['provider', 'model', 'ollama_url', 'prompt', 'input_folder']
    missing_keys = [key for key in required_keys if not final_config.get(key)]
    if missing_keys:
        logging.error(f"Missing required configuration keys: {', '.join(missing_keys)}.")
        return
    if final_config.get('provider') != 'ollama':
        logging.error("Only 'ollama' provider is supported.")
        return

    input_folder = final_config['input_folder']
    output_file = final_config['output_file']
    max_workers = final_config['max_workers']
    ollama_url = final_config['ollama_url']
    model = final_config['model']
    prompt = final_config['prompt']

    # Determine temporary file path
    output_dir = os.path.dirname(os.path.abspath(output_file))
    base_output_filename = os.path.basename(output_file)
    temp_output_file = os.path.join(output_dir, base_output_filename + TEMP_OUTPUT_SUFFIX)

    # Ensure output directory exists for temp file
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Cannot create output directory {output_dir}: {e}")
            return

    if not os.path.isdir(input_folder):
        logging.error(f"Input folder not found: {input_folder}")
        return

    all_pdf_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    if not all_pdf_files:
        logging.warning(f"No PDF files found in {input_folder}")
        # If no input files, still run cleaning in case a temp file exists from previous run
        clean_jsonl_output(temp_output_file, output_file)
        return

    # --- Session Management ---
    processed_files_set = load_processed_files()
    pdf_files_to_process = [p for p in all_pdf_files if os.path.basename(p) not in processed_files_set]
    skipped_count = len(all_pdf_files) - len(pdf_files_to_process)
    if skipped_count > 0:
        logging.info(f"Skipping {skipped_count} files already processed in previous sessions.")

    if not pdf_files_to_process:
        logging.info("No new PDF files to process.")
        # Run cleaning on existing temp file if no new files
        clean_jsonl_output(temp_output_file, output_file)
        return

    logging.info(f"Found {len(pdf_files_to_process)} new PDF files to process.")

    try:
        ollama_client = ollama.Client(host=ollama_url)
        ollama_client.list() # Check connection
        logging.info(f"Ollama client initialized and server reachable at {ollama_url}.")
    except Exception as e:
         logging.error(f"Failed to initialize or connect to Ollama client at {ollama_url}: {e}")
         return

    # --- Processing with Tqdm ---
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf_path, ollama_client, model, prompt): pdf_path for pdf_path in pdf_files_to_process}

        # Use tqdm for progress bar
        for future in tqdm(as_completed(future_to_pdf), total=len(pdf_files_to_process), desc="Processing PDFs"):
            pdf_path = future_to_pdf[future]
            pdf_filename = os.path.basename(pdf_path)
            try:
                _, result_data = future.result()
                # --- Incremental Saving ---
                if append_result_to_jsonl(result_data, temp_output_file):
                    # --- Session Saving ---
                    # Only save to session if writing to temp file was successful
                    if result_data.get("status") != "failed": # Also check if PDF processing failed
                         save_processed_file(pdf_filename)
                    else:
                         logging.warning(f"PDF processing failed for {pdf_filename}, not adding to session log.")
                else:
                    logging.error(f"Failed to save result for {pdf_filename} incrementally, will not be added to session log.")

            except Exception as exc:
                logging.error(f'{pdf_filename} generated an exception during future processing: {exc}')
                # Optionally write an error record to the temp file
                error_data = {"pdf_filename": pdf_filename, "status": "failed", "error": f"Concurrency execution error: {exc}", "pages": []}
                append_result_to_jsonl(error_data, temp_output_file) # Try to log the failure

    logging.info("All PDF processing tasks submitted and completed.")

    # --- Final Cleaning ---
    clean_jsonl_output(temp_output_file, output_file)

if __name__ == "__main__":
    main()
