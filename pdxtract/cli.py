"""Command-line interface for PDXTRACT."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from pdxtract.config import DEFAULT_CONFIG_FILE, Config, load_config
from pdxtract.core.extractor import build_extractor_registry
from pdxtract.core.pipeline import ProcessingPipeline
from pdxtract.models import PdfResult
from pdxtract.output.csv_writer import write_csv, write_txt
from pdxtract.output.json_writer import (
    append_jsonl,
    finalize_json,
    read_jsonl,
    remove_temp_file,
)
from pdxtract.output.session import load_processed_files, save_processed_file
from pdxtract.providers.base import BaseOCRProvider
from pdxtract.providers.registry import get_provider, list_available_providers
from pdxtract.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
console = Console()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdxtract",
        description="Professional PDF extraction via OCR + deterministic regex.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help="Path to config.json",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # process command
    process_parser = subparsers.add_parser("process", help="Process PDFs (default)")
    process_parser.add_argument("--provider", help="OCR backend")
    process_parser.add_argument("--model", help="Model name or HF repo id")
    process_parser.add_argument("--strategy", help="Extraction strategy")
    process_parser.add_argument("--extractors", help="Comma-separated extractor names")
    process_parser.add_argument("--ollama-url", dest="ollama_url", help="Ollama API URL")
    process_parser.add_argument(
        "--llama-cpp-url", dest="llama_cpp_url", help="llama.cpp server URL"
    )
    process_parser.add_argument("--input-folder", dest="input_folder", help="Input PDF folder")
    process_parser.add_argument("--output-file", dest="output_file", help="Final JSON output path")
    process_parser.add_argument("--output-csv", dest="output_csv", help="Optional CSV output path")
    process_parser.add_argument("--output-txt", dest="output_txt", help="Optional TXT output path")
    process_parser.add_argument(
        "--session-file", dest="session_file", help="Resume session log path"
    )
    process_parser.add_argument("--max-workers", dest="max_workers", type=int, help="Concurrency")
    process_parser.add_argument("--page-dpi", dest="page_dpi", type=int, help="Rendering DPI")
    process_parser.add_argument(
        "--include-ocr-text",
        dest="include_ocr_text",
        action="store_true",
        help="Include raw OCR text in final JSON",
    )
    process_parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Ignore session log and reprocess all files",
    )
    process_parser.add_argument(
        "--keep-temp",
        dest="keep_temp",
        action="store_true",
        default=False,
        help="Keep the incremental JSONL temp file after finalizing",
    )

    # list-models command
    list_parser = subparsers.add_parser("list-models", help="List models for configured provider")
    list_parser.add_argument("--provider", help="Provider to query")
    list_parser.add_argument("--ollama-url", dest="ollama_url", help="Ollama API URL")
    list_parser.add_argument("--llama-cpp-url", dest="llama_cpp_url", help="llama.cpp server URL")

    return parser


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Override config values with CLI arguments."""
    data = config.model_dump()
    overrides = {
        "provider": args.provider,
        "model": args.model,
        "strategy": args.strategy,
        "ollama_url": args.ollama_url,
        "llama_cpp_url": args.llama_cpp_url,
        "input_folder": args.input_folder,
        "output_file": args.output_file,
        "output_csv": args.output_csv,
        "output_txt": args.output_txt,
        "session_file": args.session_file,
        "max_workers": args.max_workers,
        "page_dpi": args.page_dpi,
        "include_ocr_text": args.include_ocr_text,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    if args.extractors:
        data["extractors"] = [e.strip() for e in args.extractors.split(",") if e.strip()]

    return Config.model_validate(data)


def discover_pdfs(input_folder: str) -> list[str]:
    folder = Path(input_folder)
    if not folder.is_dir():
        logger.error("Input folder not found: %s", folder)
        return []
    pdfs = sorted([str(p) for p in folder.iterdir() if p.suffix.lower() == ".pdf"])
    logger.info("Discovered %s PDF files in %s", len(pdfs), folder)
    return pdfs


def process_single_pdf(
    pdf_path: str,
    provider: BaseOCRProvider,
    extractor_registry: Any,
    config: Config,
) -> PdfResult:
    pipeline = ProcessingPipeline(provider, extractor_registry, config)
    return pipeline.process_pdf(pdf_path)


def cmd_process(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        config = apply_cli_overrides(config, args)
    except Exception as exc:
        logger.error("Invalid configuration: %s", exc)
        return 1

    logger.info(
        "PDXTRACT v2 — provider=%s model=%s strategy=%s",
        config.provider,
        config.model,
        config.strategy,
    )
    logger.info("Extractors: %s", ", ".join(config.extractors))

    pdfs = discover_pdfs(config.input_folder)
    if not pdfs:
        console.print("[yellow]No PDF files found.[/yellow]")
        return 0

    processed = load_processed_files(config.session_file) if args.resume else set()
    pdfs_to_process = [p for p in pdfs if Path(p).name not in processed]
    skipped = len(pdfs) - len(pdfs_to_process)
    if skipped:
        logger.info("Skipping %s already processed files", skipped)
    if not pdfs_to_process:
        console.print("[green]All PDFs already processed.[/green]")
        _finalize(config, args.keep_temp)
        return 0

    try:
        provider = get_provider(config)
    except Exception as exc:
        logger.error("Could not initialize provider '%s': %s", config.provider, exc)
        return 1

    if not provider.health_check():
        logger.error("Provider '%s' health check failed", config.provider)
        return 1

    max_workers = config.max_workers
    if not provider.is_thread_safe and max_workers > 1:
        logger.warning("Provider '%s' is not thread-safe; forcing max_workers=1", config.provider)
        max_workers = 1

    extractor_registry = build_extractor_registry(config.custom_regex)
    temp_file = config.temp_output_file()

    failed_pdfs: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Processing {len(pdfs_to_process)} PDFs with {config.provider}...",
            total=len(pdfs_to_process),
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(
                    process_single_pdf, pdf_path, provider, extractor_registry, config
                ): pdf_path
                for pdf_path in pdfs_to_process
            }
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                pdf_name = Path(pdf_path).name
                try:
                    result = future.result()
                    append_jsonl(result, temp_file)
                    if result.status == "success":
                        save_processed_file(config.session_file, pdf_name)
                    else:
                        failed_pdfs.append(pdf_name)
                        logger.warning("PDF failed: %s — %s", pdf_name, result.error)
                except Exception as exc:
                    failed_pdfs.append(pdf_name)
                    logger.error("PDF crashed: %s — %s", pdf_name, exc)
                progress.advance(task)

    provider.close()

    if failed_pdfs:
        console.print(
            f"[yellow]{len(failed_pdfs)} PDF(s) failed/will be retried next run.[/yellow]"
        )

    _finalize(config, args.keep_temp)
    return 0


def _finalize(config: Config, keep_temp: bool) -> None:
    temp_file = config.temp_output_file()
    success = finalize_json(
        temp_file,
        config.output_file,
        include_ocr_text=config.include_ocr_text,
    )
    if not success:
        return

    records = read_jsonl(temp_file)

    if config.output_csv:
        write_csv(records, config.output_csv, config.extractors)

    if config.output_txt:
        write_txt(records, config.output_txt, config.extractors)

    if not keep_temp:
        remove_temp_file(temp_file)


def cmd_list_models(args: argparse.Namespace) -> int:
    config_data = {
        "provider": args.provider or "ollama",
        "ollama_url": args.ollama_url or "http://127.0.0.1:11434",
        "llama_cpp_url": args.llama_cpp_url or "http://127.0.0.1:8080",
    }
    config = Config.model_validate(config_data)

    available = list_available_providers()
    console.print(f"[bold]Available providers:[/bold] {', '.join(available)}")

    try:
        provider = get_provider(config)
    except Exception as exc:
        logger.error("Could not initialize provider '%s': %s", config.provider, exc)
        return 1

    if not provider.health_check():
        console.print(f"[red]Provider '{config.provider}' is not reachable.[/red]")
        return 1

    models = provider.list_models()
    if models is None:
        console.print(
            f"[yellow]Provider '{config.provider}' does not support model listing.[/yellow]"
        )
    elif not models:
        console.print(f"[yellow]No models found for provider '{config.provider}'.[/yellow]")
    else:
        console.print(f"[bold]Models for {config.provider}:[/bold]")
        for m in models:
            console.print(f"  • {m}")

    provider.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    if argv is None:
        argv = sys.argv[1:]

    # Allow top-level flags without requiring the 'process' subcommand.
    if argv and argv[0] not in ("process", "list-models", "--help", "-h"):
        argv.insert(0, "process")

    args = parser.parse_args(argv)

    setup_logging(use_rich=True)

    if args.command is None or args.command == "process":
        return cmd_process(args)
    if args.command == "list-models":
        return cmd_list_models(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
