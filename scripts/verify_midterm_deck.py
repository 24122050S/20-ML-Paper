"""Kiểm tra số slide, dung lượng và speaker notes của PPTX giữa kỳ."""

import argparse
from pathlib import Path
import posixpath
import sys
from urllib.parse import unquote
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile


DECK = Path(__file__).resolve().parents[1] / "slides" / "ACIL_Nhom3_GiuaKy.pptx"
PRESENTATION_PART = "ppt/presentation.xml"
MAX_BYTES = 20_000_000
MAX_NOTE_WORDS = 500


def relationships(archive, part, kind):
    """Đọc liên kết nội bộ trong PPTX.

    Tham số ``archive`` là gói ZIP, ``part`` là đường dẫn nguồn và ``kind``
    là loại liên kết. Kết quả ánh xạ ID liên kết tới đường dẫn đích trong gói.
    """
    folder, filename = posixpath.split(part)
    rels_path = posixpath.join(folder, "_rels", f"{filename}.rels")
    try:
        root = ET.fromstring(archive.read(rels_path))
    except KeyError:
        return {}

    targets = {}
    for rel in root.findall("{*}Relationship"):
        if rel.attrib["Type"].rsplit("/", 1)[-1] != kind or rel.get("TargetMode") == "External":
            continue
        target = unquote(rel.attrib["Target"])
        targets[rel.attrib["Id"]] = posixpath.normpath(posixpath.join(folder, target)).lstrip("/")
    return targets


def note_word_count(root):
    """Nhận phần tử XML gốc của notes và trả số từ trong ô nội dung.

    Các đoạn văn và ngắt dòng tách từ, còn các run định dạng được nối liền.
    """
    count = 0
    for shape in root.findall(".//{*}sp"):
        placeholder = shape.find("{*}nvSpPr/{*}nvPr/{*}ph")
        if placeholder is None or placeholder.get("type") != "body":
            continue
        for paragraph in shape.findall("{*}txBody/{*}p"):
            text = []
            for element in paragraph.iter():
                tag = element.tag.rsplit("}", 1)[-1]
                if tag == "t":
                    text.append(element.text or "")
                elif tag == "br":
                    text.append(" ")
            count += len("".join(text).split())
    return count


def word_counts_by_slide(archive):
    """Nhận gói ZIP của PPTX và trả số từ notes theo thứ tự trình chiếu."""
    presentation = ET.fromstring(archive.read(PRESENTATION_PART))
    slides = relationships(archive, PRESENTATION_PART, "slide")
    counts = []
    for slide_id in presentation.findall("{*}sldIdLst/{*}sldId"):
        # Thuộc tính id không có namespace là ID số, khác với r:id của liên kết.
        rel_id = next((value for key, value in slide_id.attrib.items() if key.endswith("}id")), None)
        slide = slides[rel_id]
        archive.getinfo(slide)
        notes = list(relationships(archive, slide, "notesSlide").values())
        if len(notes) > 1:
            raise ValueError(f"{slide} links to more than one notes slide")
        count = note_word_count(ET.fromstring(archive.read(notes[0]))) if notes else 0
        counts.append(count)
    return counts


def main():
    """Nhận đường dẫn PPTX qua CLI, báo lỗi và trả 0 khi đạt hoặc 1 khi lỗi."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck", nargs="?", type=Path, default=DECK, help="PPTX file to check")
    deck = parser.parse_args().deck
    try:
        if deck.suffix.lower() != ".pptx":
            raise ValueError("expected a .pptx file")
        size = deck.stat().st_size
        with ZipFile(deck) as archive:
            counts = word_counts_by_slide(archive)
    except (OSError, BadZipFile, ET.ParseError, KeyError, ValueError) as exc:
        print(f"{deck}: {exc}", file=sys.stderr)
        return 1

    errors = []
    if not 20 <= len(counts) <= 40:
        errors.append(f"{len(counts)} slides; expected 20–40")
    if size >= MAX_BYTES:
        errors.append(f"{size:,} bytes; must be under {MAX_BYTES:,}")
    for index, count in enumerate(counts, start=1):
        if count == 0:
            errors.append(f"slide {index} has no speaker notes")
        elif count > MAX_NOTE_WORDS:
            errors.append(f"slide {index} has {count} words in speaker notes; limit is {MAX_NOTE_WORDS}")

    print(
        f"{deck.name}: {len(counts)} slides, {size:,} bytes, "
        f"{sum(count > 0 for count in counts)} slides with notes, "
        f"longest notes {max(counts, default=0)} words."
    )
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
