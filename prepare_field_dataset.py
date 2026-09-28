"""Prepare PlantWild v2 research data with reproducible, exact-duplicate-free splits."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='datasets/external/plantwild_v2.zip')
    parser.add_argument('--output', default='datasets/processed/plantwild_v2')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise ValueError('Output already exists. Choose a new empty directory.')
    groups, seen = defaultdict(list), {}
    duplicates, corrupt, conflicts = [], [], []
    with ZipFile(args.archive) as archive:
        members = [n for n in archive.namelist() if Path(n).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}]
        for index, name in enumerate(members):
            parts = Path(name).parts
            if len(parts) != 3 or parts[0] != 'plantwild_v2' or any(p in {'.', '..'} for p in parts):
                raise ValueError('Unexpected archive path: ' + name)
            content = archive.read(name)
            try:
                with Image.open(BytesIO(content)) as image:
                    image.verify()
            except Exception as error:
                corrupt.append({'name': name, 'error': str(error)})
                continue
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen:
                duplicates.append([seen[digest], name])
                if Path(seen[digest]).parts[1] != parts[1]:
                    conflicts.append([seen[digest], name])
                continue
            seen[digest] = name
            groups[parts[1]].append((name, digest))
            if (index + 1) % 2000 == 0:
                print('Checked', index + 1, flush=True)
        # Ambiguous duplicate labels must not enter any split.
        excluded = {name for pair in conflicts for name in pair}
        groups = {label: sorted(item for item in entries if item[0] not in excluded) for label, entries in groups.items()}
        insufficient = {label: len(entries) for label, entries in groups.items() if len(entries) < 7}
        classes, rng, manifest, counts = sorted(label for label in groups if label not in insufficient), random.Random(42), [], defaultdict(int)
        for label in classes:
            entries = sorted(item for item in groups[label] if item[0] not in excluded)
            rng.shuffle(entries)
            if len(entries) < 7:
                raise ValueError('Too few independent images for ' + label)
            holdout = max(1, round(len(entries) * .15))
            for index, (name, digest) in enumerate(entries):
                split = 'test' if index < holdout else 'valid' if index < 2 * holdout else 'train'
                destination = root / split / label / (digest + Path(name).suffix.lower())
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
                counts[split] += 1
                manifest.append({'path': destination.relative_to(root).as_posix(), 'source': name,
                                 'sha256': digest, 'class': label, 'split': split})
    (root / 'classes.json').write_text(json.dumps(classes, indent=2), encoding='utf-8')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    report = {'dataset': 'PlantWild v2', 'source': 'https://huggingface.co/datasets/uqtwei2/PlantWild',
              'license': 'CC BY-NC-ND 4.0. Research candidate; review redistribution and commercial rights.',
              'classes': len(classes), 'counts': dict(counts), 'exact_duplicates_removed': duplicates,
              'conflicting_labels_excluded': conflicts, 'corrupt_images': corrupt, 'seed': 42,
              'classes_excluded_for_too_few_images': insufficient,
              'note': 'Random stratified benchmark; near duplicates and source leakage remain unaudited. Not independent farmer-photo accuracy.'}
    (root / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ['exact_duplicates_removed', 'conflicting_labels_excluded', 'corrupt_images']}), flush=True)


if __name__ == '__main__':
    main()
