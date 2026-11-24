"""Locate the 24 redistributable tracks in the official Medium archive for HTTP range reads."""
import hashlib
import json
import struct
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CATALOGUE = HERE / "public/listening-catalogue.json"


def main():
    """Save verified byte ranges and content hashes without copying any audio."""
    archive = HERE.parent / "data/medium/data/raw/fma_medium.zip"
    entries = {}
    with zipfile.ZipFile(archive) as package, archive.open("rb") as source:
        for track in json.loads(CATALOGUE.read_text(encoding="utf8")):
            tid = f"{track['track_id']:06d}"
            info = package.getinfo(f"fma_medium/{tid[:3]}/{tid}.mp3")
            assert info.compress_type == zipfile.ZIP_BZIP2
            source.seek(info.header_offset)
            header = source.read(30)
            assert header[:4] == b"PK\x03\x04"
            name_size, extra_size = struct.unpack_from("<HH", header, 26)
            entries[str(track["track_id"])] = {
                "start": info.header_offset + 30 + name_size + extra_size,
                "compressed": info.compress_size,
                "size": info.file_size,
                "sha256": hashlib.sha256(package.read(info)).hexdigest(),
            }
    (HERE / "sample-index.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
