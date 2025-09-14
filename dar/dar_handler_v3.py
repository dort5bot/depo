# handlers/dar_handler.py
"""
DENENMEDİ
✅ Özellikler:
 Aiogram 3.x uyumlu
/dar → Proje ağacını mesaj olarak (çok büyükse .txt dosyası olarak) gönderir.
/dar k → Tüm Python dosyalarında @router.message(Command(...)) komutlarını bulur ve alfabetik listeler.
/dar t → Proje ağacını .txt dosyası olarak gönderir.
/dar Z → Tüm projeyi .zip dosyası olarak gönderir.
"""
import os
import re
import zipfile
from pathlib import Path
from aiogram import Router
from aiogram.types import Message, FSInputFile

router = Router()

# Proje kök dizini (bu dosyanın 2 üst klasörü genelde proje root olur)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_tree(directory: Path, prefix: str = "") -> str:
    """Proje ağaç yapısını döndürür."""
    tree = ""
    entries = sorted(directory.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        tree += f"{prefix}{connector}{entry.name}\n"
        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            tree += get_project_tree(entry, prefix + extension)
    return tree


def scan_commands_in_project() -> str:
    """Tüm .py dosyalarını tarar ve @router.message(Command(...)) komutlarını listeler."""
    commands = []
    pattern = re.compile(r'@router\.message\(.*Command\(["\']([^"\']+)["\']')
    for pyfile in PROJECT_ROOT.rglob("*.py"):
        if pyfile.name.startswith("__"):
            continue
        try:
            with pyfile.open("r", encoding="utf-8") as f:
                content = f.read()
            matches = pattern.findall(content)
            for cmd in matches:
                commands.append(f"{pyfile.relative_to(PROJECT_ROOT)} → /{cmd}")
        except Exception:
            continue
    if not commands:
        return "❌ Hiç komut bulunamadı."
    return "📜 Bulunan Komutlar:\n" + "\n".join(sorted(commands))


def create_txt_tree() -> Path:
    """Proje ağacını .txt dosyasına kaydeder."""
    txt_path = PROJECT_ROOT / "project_tree.txt"
    tree = f"📂 Proje Ağacı: {PROJECT_ROOT.name}\n\n"
    tree += get_project_tree(PROJECT_ROOT)
    txt_path.write_text(tree, encoding="utf-8")
    return txt_path


def create_zip_project() -> Path:
    """Tüm projeyi zip olarak arşivler."""
    zip_path = PROJECT_ROOT / "project_backup.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in PROJECT_ROOT.rglob("*"):
            if file.is_file():
                zipf.write(file, file.relative_to(PROJECT_ROOT))
    return zip_path


@router.message(lambda m: m.text and m.text.strip() == "/dar")
async def cmd_dar(message: Message):
    """Proje ağaç yapısını mesaj olarak gönderir."""
    tree = f"📂 Proje Ağacı: {PROJECT_ROOT.name}\n\n"
    tree += get_project_tree(PROJECT_ROOT)
    if len(tree) > 4000:  # Telegram mesaj limiti
        txt_path = create_txt_tree()
        await message.answer_document(FSInputFile(txt_path))
    else:
        await message.answer(f"<pre>{tree}</pre>", parse_mode="HTML")


@router.message(lambda m: m.text and m.text.strip() == "/dar k")
async def cmd_dar_k(message: Message):
    """Tüm komutları listeler."""
    result = scan_commands_in_project()
    await message.answer(f"<pre>{result}</pre>", parse_mode="HTML")


@router.message(lambda m: m.text and m.text.strip() == "/dar t")
async def cmd_dar_t(message: Message):
    """Proje ağacını .txt dosyası olarak gönderir."""
    txt_path = create_txt_tree()
    await message.answer_document(FSInputFile(txt_path))


@router.message(lambda m: m.text and m.text.strip() == "/dar Z")
async def cmd_dar_Z(message: Message):
    """Tüm proje klasörünü zip dosyası olarak gönderir."""
    zip_path = create_zip_project()
    await message.answer_document(FSInputFile(zip_path))
