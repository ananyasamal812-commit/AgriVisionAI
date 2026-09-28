"""Start the local API and web interface; Ctrl+C stops both."""
import os
import argparse
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent

def choose_port(requested, preferred):
    ports = [requested] if requested is not None else range(preferred, preferred + 100)
    for port in ports:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port available starting at {requested or preferred}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-port', type=int)
    parser.add_argument('--web-port', type=int)
    args = parser.parse_args()
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if not npm:
        raise SystemExit("Node.js/npm is missing. Install Node.js, then reopen your terminal.")
    if not (ROOT / "frontend/node_modules").is_dir():
        raise SystemExit("Run npm run install:frontend first.")
    args.api_port = choose_port(args.api_port, 8000)
    args.web_port = choose_port(args.web_port, 5173)
    if args.api_port == args.web_port:
        raise SystemExit("API and frontend ports must be different.")
    children = []
    env = os.environ.copy()
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["AGRIVISION_API_PORT"] = str(args.api_port)
    env["AGRIVISION_WEB_PORT"] = str(args.web_port)
    try:
        children.append(subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(args.api_port)],
            cwd=ROOT, env=env))
        print("Loading prediction model...", flush=True)
        deadline = time.monotonic() + 180
        while True:
            if children[0].poll() is not None:
                raise RuntimeError("API failed to start. See the error above.")
            try:
                with urlopen(f"http://127.0.0.1:{args.api_port}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError("API startup timed out. Check the model and Python dependencies.")
            time.sleep(1)
        children.append(subprocess.Popen([npm, "run", "dev:web"], cwd=ROOT, env=env))
        print(f"Open http://127.0.0.1:{args.web_port}. Press Ctrl+C to stop both services.", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError("A service stopped. See its terminal output above.")
    except KeyboardInterrupt:
        print("\nStopping AgriVisionAI...")
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1
    finally:
        for child in reversed(children):
            if child.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
    return 0

if __name__ == "__main__":
    sys.exit(main())

