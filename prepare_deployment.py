"""Export only the files needed for the HF API and Vercel frontend."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent


def main():
    target = ROOT / "outputs/deployment"
    target.mkdir(parents=True, exist_ok=True)
    backend = ["Dockerfile", ".dockerignore", "requirements-api.txt", "app.py",
               "configs/config.yaml", "configs/classes.json", "data/disease_info.json", "checkpoints/best_model.pth"]
    backend += [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")]
    backend += [p.relative_to(ROOT).as_posix() for folder in ["assets/fonts", "data/locales"] for p in (ROOT / folder).rglob("*") if p.is_file()]
    with ZipFile(target / "huggingface-api.zip", "w", ZIP_DEFLATED) as archive:
        for name in backend:
            archive.write(ROOT / name, name)
        archive.write(ROOT / "deployment/SPACE_README.md", "README.md")
        archive.writestr(".gitattributes", "*.pth filter=lfs diff=lfs merge=lfs -text\n")
    frontend = ["package.json", "package-lock.json", "index.html", "vite.config.js", "vercel.json", ".env.example"]
    frontend += [p.relative_to(ROOT / "frontend").as_posix() for folder in ["src", "public", "scripts"]
                 for p in (ROOT / "frontend" / folder).rglob("*") if p.is_file() and p.name != "Evaluation.jsx"]
    with ZipFile(target / "vercel-frontend.zip", "w", ZIP_DEFLATED) as archive:
        for name in frontend:
            archive.write(ROOT / "frontend" / name, name)
    print(f"Deployment archives: {target}")


if __name__ == "__main__":
    main()
