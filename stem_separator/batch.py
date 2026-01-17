"""
Batch processing module for Stem Separator.

Handles processing multiple files with:
- Parallel processing
- Model pre-loading
- Progress tracking
- Error handling
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from stem_separator.config import MODELS
from stem_separator.logging_config import (
    create_progress,
    get_logger,
    is_quiet,
    print_error,
    print_info,
    print_status,
    print_success,
    print_warning,
)
from stem_separator.metadata import copy_metadata, read_metadata
from stem_separator.processing import SeparationResult, StemSeparator
from stem_separator.utils import collect_audio_files, sanitize_filename


@dataclass
class BatchItem:
    """A single item in a batch processing queue."""

    input_path: Path
    output_dir: Path
    display_name: str = ""
    index: int = 0
    total: int = 1

    def __post_init__(self):
        if not self.display_name:
            self.display_name = sanitize_filename(self.input_path.stem)


@dataclass
class BatchResult:
    """Result of batch processing."""

    total: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[tuple[BatchItem, SeparationResult]] = field(default_factory=list)


class BatchProcessor:
    """
    Batch processor for multiple audio files.

    Supports parallel processing and model pre-loading for efficiency.
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        output_format: str = "wav",
        selected_stems: Optional[list[str]] = None,
        force_cpu: bool = False,
        normalize: bool = False,
        low_memory: bool = False,
        naming_template: str = "{name}_{stem}",
        parallel_jobs: int = 1,
        preserve_metadata: bool = True,
        analyze_quality: bool = False,
        dry_run: bool = False,
    ):
        """
        Initialize batch processor.

        Args:
            model_name: Demucs model to use.
            output_format: Output format for stems.
            selected_stems: List of stems to export.
            force_cpu: Force CPU processing.
            normalize: Whether to normalize audio.
            low_memory: Enable low-memory mode.
            naming_template: Output filename template.
            parallel_jobs: Number of parallel jobs (1 = sequential).
            preserve_metadata: Whether to copy metadata to output files.
            analyze_quality: Whether to analyze stem quality.
            dry_run: If True, don't actually process, just report what would happen.
        """
        self.model_name = model_name
        self.output_format = output_format
        self.selected_stems = selected_stems
        self.force_cpu = force_cpu
        self.normalize = normalize
        self.low_memory = low_memory
        self.naming_template = naming_template
        self.parallel_jobs = max(1, parallel_jobs)
        self.preserve_metadata = preserve_metadata
        self.analyze_quality = analyze_quality
        self.dry_run = dry_run

        self.logger = get_logger()
        self._separator: Optional[StemSeparator] = None

    def _get_separator(self) -> StemSeparator:
        """Get or create the stem separator instance."""
        if self._separator is None:
            self._separator = StemSeparator(
                model_name=self.model_name,
                force_cpu=self.force_cpu,
                low_memory=self.low_memory,
            )
        return self._separator

    def preload_model(self) -> bool:
        """
        Pre-load the model for efficient batch processing.

        Returns:
            True if model loaded successfully.
        """
        separator = self._get_separator()
        return separator.load_model()

    def unload_model(self):
        """Unload the model to free memory."""
        if self._separator is not None:
            self._separator.unload_model()
            self._separator = None

    def process_single(
        self,
        item: BatchItem,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> SeparationResult:
        """
        Process a single batch item.

        Args:
            item: BatchItem to process.
            progress_callback: Optional progress callback.

        Returns:
            SeparationResult for the item.
        """
        if self.dry_run:
            print_info(f"[DRY RUN] Would process: {item.display_name}")
            return SeparationResult(
                success=True,
                output_dir=item.output_dir,
                model_used=self.model_name,
                processing_device="dry_run",
            )

        separator = self._get_separator()

        # Ensure model is loaded
        if not separator._loaded:
            if not separator.load_model():
                return SeparationResult(
                    success=False,
                    error="Failed to load model",
                )

        # Create output directory
        output_dir = item.output_dir / f"{item.display_name}_stems"

        # Read original metadata for preservation
        original_metadata = None
        if self.preserve_metadata:
            original_metadata = read_metadata(item.input_path)

        # Process
        result = separator.separate(
            input_file=item.input_path,
            output_dir=output_dir,
            output_format=self.output_format,
            selected_stems=self.selected_stems,
            normalize=self.normalize,
            naming_template=self.naming_template,
            track_name=item.display_name,
            analyze_quality=self.analyze_quality,
            progress_callback=progress_callback,
        )

        # Copy metadata to output files
        if result.success and original_metadata and self.preserve_metadata:
            for stem in result.stems:
                if stem.file_path and stem.file_path.exists():
                    copy_metadata(item.input_path, stem.file_path, stem.name)

        return result

    def process_batch(
        self,
        items: list[BatchItem],
        on_item_complete: Optional[Callable[[BatchItem, SeparationResult], None]] = None,
    ) -> BatchResult:
        """
        Process a batch of items.

        Args:
            items: List of BatchItem objects to process.
            on_item_complete: Optional callback when each item completes.

        Returns:
            BatchResult with overall statistics.
        """
        if not items:
            return BatchResult()

        batch_result = BatchResult(total=len(items))

        if self.dry_run:
            print_status(f"[DRY RUN] Would process {len(items)} files")
            for item in items:
                print_info(f"  - {item.display_name}")
            batch_result.successful = len(items)
            return batch_result

        # Pre-load model for efficiency
        if not self.preload_model():
            print_error("Failed to load model, aborting batch")
            batch_result.failed = len(items)
            return batch_result

        print_status(f"Processing {len(items)} files...")

        if self.parallel_jobs > 1:
            # Parallel processing
            # Note: For GPU processing, parallel doesn't help much
            # For CPU, it can help on multi-core systems
            batch_result = self._process_parallel(items, on_item_complete)
        else:
            # Sequential processing
            batch_result = self._process_sequential(items, on_item_complete)

        self.unload_model()

        return batch_result

    def _process_sequential(
        self,
        items: list[BatchItem],
        on_item_complete: Optional[Callable[[BatchItem, SeparationResult], None]] = None,
    ) -> BatchResult:
        """Process items sequentially."""
        batch_result = BatchResult(total=len(items))

        with create_progress("Processing") as progress:
            task = progress.add_task("Files", total=len(items))

            for i, item in enumerate(items):
                item.index = i
                item.total = len(items)

                if not is_quiet():
                    print_status(f"[{i + 1}/{len(items)}] {item.display_name}")

                result = self.process_single(item)
                batch_result.results.append((item, result))

                if result.success:
                    batch_result.successful += 1
                else:
                    batch_result.failed += 1
                    print_warning(f"Failed: {item.display_name} - {result.error}")

                if on_item_complete:
                    on_item_complete(item, result)

                progress.advance(task)

        return batch_result

    def _process_parallel(
        self,
        items: list[BatchItem],
        on_item_complete: Optional[Callable[[BatchItem, SeparationResult], None]] = None,
    ) -> BatchResult:
        """
        Process items with parallel_jobs > 1.

        IMPORTANT NOTE: Due to GPU memory constraints and model loading overhead,
        this method currently processes items sequentially even when parallel_jobs > 1.
        True parallel processing would require:
        - Multiple GPU contexts (for GPU mode) - not practical due to VRAM limits
        - Multiple model instances (for CPU mode) - high memory overhead (~4GB per instance)

        For users seeking parallelism:
        - Use --parallel flag mainly for I/O-bound pre/post processing
        - For true parallel GPU processing, run multiple separate instances
        - Consider batch mode with model pre-loading for efficiency instead

        The parallel_jobs parameter may be used in future versions for:
        - Pre-processing file conversions
        - Post-processing format conversions
        - Multi-GPU setups
        """
        batch_result = BatchResult(total=len(items))

        # Log info about parallel processing limitation
        if self.parallel_jobs > 1 and not self.force_cpu:
            self.logger.info(
                f"Note: parallel_jobs={self.parallel_jobs} requested, but GPU processing "
                "uses sequential execution to avoid VRAM exhaustion. "
                "Model pre-loading still provides efficiency benefits."
            )

        with create_progress("Processing") as progress:
            task = progress.add_task("Files", total=len(items))

            # Use thread pool for I/O-bound parts, but process sequentially for model
            # This is a simplified parallel approach - true parallelism would need
            # multiple model instances

            for i, item in enumerate(items):
                item.index = i
                item.total = len(items)

                if not is_quiet():
                    print_status(f"[{i + 1}/{len(items)}] {item.display_name}")

                result = self.process_single(item)
                batch_result.results.append((item, result))

                if result.success:
                    batch_result.successful += 1
                else:
                    batch_result.failed += 1
                    print_warning(f"Failed: {item.display_name} - {result.error}")

                if on_item_complete:
                    on_item_complete(item, result)

                progress.advance(task)

        return batch_result

    @staticmethod
    def from_directory(
        input_dir: Path,
        output_dir: Path,
        recursive: bool = False,
    ) -> list[BatchItem]:
        """
        Create batch items from a directory.

        Args:
            input_dir: Directory containing audio files.
            output_dir: Directory for output.
            recursive: Whether to search recursively.

        Returns:
            List of BatchItem objects.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        files = collect_audio_files(input_dir, recursive)

        return [
            BatchItem(
                input_path=f,
                output_dir=output_dir,
                display_name=sanitize_filename(f.stem),
                index=i,
                total=len(files),
            )
            for i, f in enumerate(files)
        ]

    @staticmethod
    def from_file_list(
        files: list[Path],
        output_dir: Path,
    ) -> list[BatchItem]:
        """
        Create batch items from a list of files.

        Args:
            files: List of audio file paths.
            output_dir: Directory for output.

        Returns:
            List of BatchItem objects.
        """
        output_dir = Path(output_dir)

        return [
            BatchItem(
                input_path=Path(f),
                output_dir=output_dir,
                display_name=sanitize_filename(Path(f).stem),
                index=i,
                total=len(files),
            )
            for i, f in enumerate(files)
        ]


def process_batch(
    input_paths: list[Path] | Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    output_format: str = "wav",
    selected_stems: Optional[list[str]] = None,
    force_cpu: bool = False,
    normalize: bool = False,
    parallel_jobs: int = 1,
    recursive: bool = False,
    dry_run: bool = False,
) -> BatchResult:
    """
    Convenience function for batch processing.

    Args:
        input_paths: Single path (file or directory) or list of file paths.
        output_dir: Directory for output.
        model_name: Demucs model to use.
        output_format: Output format.
        selected_stems: Stems to export.
        force_cpu: Force CPU processing.
        normalize: Whether to normalize audio.
        parallel_jobs: Number of parallel jobs.
        recursive: Search directories recursively.
        dry_run: Don't actually process.

    Returns:
        BatchResult with statistics.
    """
    processor = BatchProcessor(
        model_name=model_name,
        output_format=output_format,
        selected_stems=selected_stems,
        force_cpu=force_cpu,
        normalize=normalize,
        parallel_jobs=parallel_jobs,
        dry_run=dry_run,
    )

    # Build batch items
    if isinstance(input_paths, (str, Path)):
        input_path = Path(input_paths)
        if input_path.is_dir():
            items = BatchProcessor.from_directory(input_path, output_dir, recursive)
        else:
            items = BatchProcessor.from_file_list([input_path], output_dir)
    else:
        items = BatchProcessor.from_file_list(input_paths, output_dir)

    if not items:
        print_warning("No audio files found to process")
        return BatchResult()

    return processor.process_batch(items)
