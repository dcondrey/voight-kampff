"""Download non-quantized ONNX model from Modal volume.

Usage:
    modal run download_onnx.py
"""

import logging
from pathlib import Path

import modal

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

app = modal.App("vk-download-onnx")
volume = modal.Volume.from_name("vk-models")
MOUNT_PATH = "/vol"


@app.function(volumes={MOUNT_PATH: volume})
def get_onnx_files():
    import logging

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    onnx_dir = Path(f"{MOUNT_PATH}/deberta_onnx")
    files = {}
    for f in onnx_dir.rglob("*"):
        if f.is_file():
            rel = f.relative_to(onnx_dir)
            files[str(rel)] = f.read_bytes()
            log.info("  %s (%.1f MB)", rel, f.stat().st_size / 1e6)
    return files


@app.local_entrypoint()
def main():
    output_dir = Path("models/deberta_onnx")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Downloading non-quantized ONNX model...")
    files = get_onnx_files.remote()
    for rel_path, content in files.items():
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)
        log.info("  Saved %s (%.1f MB)", rel_path, len(content) / 1e6)

    log.info("Model saved to %s/", output_dir)
