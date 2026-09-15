"""
SignalScope Checkpoint Downloader & Linker
Downloads the fine-tuned PyTorch checkpoint from Google Drive and places it
into backend/weights/ and model_engine/checkpoints/.
"""

import sys
import os
import argparse
from pathlib import Path

DEFAULT_FOLDER_ID = "1KhncwZYS6CWANEZHNefzNK5OonneTdpC"

def get_target_paths():
    base_dir = Path(__file__).resolve().parents[2] # sih root
    backend_weights = base_dir / "backend" / "weights" / "signalscope_final_calibrated.pth"
    model_engine_ckpt = base_dir / "model_engine" / "checkpoints" / "signalscope_final_calibrated.pth"
    backend_weights.parent.mkdir(parents=True, exist_ok=True)
    model_engine_ckpt.parent.mkdir(parents=True, exist_ok=True)
    return backend_weights, model_engine_ckpt

def download_with_gdown(file_or_folder_id: str, is_folder: bool, output_path: Path):
    try:
        import gdown
    except ImportError:
        print("[*] Installing 'gdown' for reliable Google Drive downloads...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    if is_folder:
        print(f"[*] Downloading folder '{file_or_folder_id}' from Google Drive...")
        folder_url = f"https://drive.google.com/drive/folders/{file_or_folder_id}"
        out_dir = output_path.parent
        gdown.download_folder(url=folder_url, output=str(out_dir), quiet=False, use_cookies=False)
        # Check if any .pth file was downloaded
        pth_files = list(out_dir.glob("*.pth")) + list(out_dir.glob("**/*.pth"))
        if pth_files:
            latest_pth = pth_files[0]
            print(f"[+] Found checkpoint: {latest_pth}")
            if latest_pth != output_path:
                import shutil
                shutil.copy2(latest_pth, output_path)
                print(f"[+] Linked checkpoint to {output_path}")
            return True
        else:
            print("[!] Folder downloaded, but no .pth file detected.")
            return False
    else:
        print(f"[*] Downloading file '{file_or_folder_id}' from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_or_folder_id}"
        gdown.download(url, str(output_path), quiet=False)
        return output_path.exists()

def main():
    parser = argparse.ArgumentParser(description="Download SignalScope fine-tuned checkpoint from Google Drive.")
    parser.add_argument("--folder_id", default=DEFAULT_FOLDER_ID, help="Google Drive folder ID")
    parser.add_argument("--file_id", default=None, help="Google Drive direct file ID")
    parser.add_argument("--local_file", default=None, help="Path to an existing local .pth file to link")
    args = parser.parse_args()

    backend_weights, model_engine_ckpt = get_target_paths()

    if args.local_file:
        src = Path(args.local_file)
        if not src.exists():
            print(f"[!] File not found: {src}")
            sys.exit(1)
        import shutil
        import zipfile

        # Check if user provided a .zip file from Google Drive
        if src.suffix.lower() == ".zip":
            print(f"[*] Detected .zip archive ({src.name}). Extracting .pth checkpoint...")
            with zipfile.ZipFile(src, "r") as zf:
                pth_names = [f for f in zf.namelist() if f.endswith(".pth")]
                if not pth_names:
                    print("[!] No .pth checkpoint found inside the zip file.")
                    sys.exit(1)
                # Extract the .pth file
                target_pth = pth_names[0]
                extracted_path = backend_weights.parent / Path(target_pth).name
                zf.extract(target_pth, backend_weights.parent)
                if extracted_path != backend_weights:
                    shutil.copy2(extracted_path, backend_weights)
        else:
            shutil.copy2(src, backend_weights)
        
        shutil.copy2(backend_weights, model_engine_ckpt)
        print(f"[+] Successfully copied and connected local checkpoint:")
        print(f"    -> {backend_weights}")
        print(f"    -> {model_engine_ckpt}")
        return

    print("=" * 65)
    print(" SignalScope Checkpoint Downloader & Linker")
    print("=" * 65)
    print(f"Target Checkpoint: {backend_weights}")

    if args.file_id:
        success = download_with_gdown(args.file_id, is_folder=False, output_path=backend_weights)
    else:
        success = download_with_gdown(args.folder_id, is_folder=True, output_path=backend_weights)

    if success and backend_weights.exists():
        import shutil
        shutil.copy2(backend_weights, model_engine_ckpt)
        print("\n[SUCCESS] Checkpoint successfully connected to SignalScope!")
        print(f"  Backend:      {backend_weights}")
        print(f"  Model Engine: {model_engine_ckpt}")
    else:
        print("\n[NOTE] If Google Drive requires login or private access permissions:")
        print("  1. Open your Google Drive link in your browser:")
        print(f"     https://drive.google.com/drive/folders/{args.folder_id}")
        print("  2. Download the .pth checkpoint file.")
        print(f"  3. Place it in: {backend_weights.resolve()}")
        print(f"     Or run: python backend/scripts/download_checkpoint.py --local_file <path_to_file.pth>")

if __name__ == "__main__":
    main()
