"""Publish the clean AgriVisionAI backend package using the locally logged-in HF account."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse
from zipfile import ZipFile
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "outputs/deployment/backend-state.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["publish", "status", "cors"])
    parser.add_argument("--origin")
    args = parser.parse_args()
    api = HfApi()
    account = api.whoami()["name"]
    repo = account + "/agrivision-ai-api"
    if args.action == "publish":
        package = ROOT / "outputs/deployment/huggingface-api.zip"
        if not package.is_file():
            raise ValueError("Run prepare_deployment.py first")
        with TemporaryDirectory(prefix="hf-upload-", dir=STATE.parent) as temporary:
            target = Path(temporary).resolve()
            with ZipFile(package) as archive:
                for entry in archive.infolist():
                    if not (target / entry.filename).resolve().is_relative_to(target):
                        raise ValueError("Invalid deployment archive path")
                archive.extractall(target)
            api.create_repo(repo_id=repo, repo_type="space", space_sdk="docker",
                            private=False, space_hardware="cpu-basic", exist_ok=True,
                            space_variables=[{"key": "ENABLE_EVALUATION", "value": "false"}])
            commit = api.upload_folder(repo_id=repo, repo_type="space", folder_path=target,
                                       commit_message="Deploy AgriVisionAI with bilingual PDF reports")
            print("Uploaded commit:", commit.oid, flush=True)
    elif args.action == "cors":
        origin = (args.origin or "").rstrip("/")
        parsed = urlparse(origin)
        if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("Provide the exact HTTPS frontend origin")
        api.add_space_variable(repo_id=repo, key="ALLOWED_ORIGINS", value=origin)
    info = api.space_info(repo)
    runtime = api.get_space_runtime(repo)
    state = {"repo": repo, "space_url": "https://huggingface.co/spaces/" + repo,
             "api_url": "https://" + info.subdomain + ".hf.space" if info.subdomain else None,
             "stage": str(runtime.stage), "hardware": str(runtime.hardware)}
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
