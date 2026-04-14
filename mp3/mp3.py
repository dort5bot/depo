# pip install pandas openpyxl requests tqdm
# pip install concurrent.futures
# Bunu Python ile oldukça temiz şekilde yapabilirsin. Temel mantık:
# 
# Excel dosyasını oku (başlıklar 1. satır)
# B sütunundaki linkleri al
# İndir
# Dosyayı kaydet (kaliteli indirme için mümkünse stream + retry)
# Başarı durumunu A sütununa yaz:
# 1 = başarılı
# 0 = başarısız
# C sütunundaki açıklamayı istersen log için kullan
# 

import os
import re
import sys
import mimetypes
import requests
import pandas as pd
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DOWNLOAD_FOLDER = "downloads"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# retry session
session = requests.Session()
retries = Retry(total=5, backoff_factor=1)
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))


def normalize_path(path):
    path = path.strip().strip('"')

    # Git Bash /d/... → D:\...
    if len(path) > 2 and path[0] == "/" and path[2] == "/":
        drive = path[1].upper()
        path = f"{drive}:{path[2:]}"

    return path
    

def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))


def get_extension(url, response):
    ext = os.path.splitext(url)[1]
    if ext and len(ext) <= 5:
        return ext

    content_type = response.headers.get("content-type", "").split(";")[0]
    return mimetypes.guess_extension(content_type) or ".bin"


def download_file(url, save_base):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        with session.get(url, stream=True, timeout=30, headers=headers) as r:
            r.raise_for_status()

            ext = get_extension(url, r)
            save_path = save_base + ext

            with open(save_path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return True

    except Exception as e:
        print(f"❌ Hata: {url} -> {e}")
        return False


def process_excel(file_path):
    df = pd.read_excel(file_path)

    link_col = df.columns[1]
    desc_col = df.columns[2]

    results = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        url = row[link_col]
        desc = row[desc_col] if len(df.columns) > 2 else f"file_{i}"

        if pd.isna(url):
            results.append(0)
            continue

        file_name = clean_filename(f"{i}_{desc}")
        save_base = os.path.join(DOWNLOAD_FOLDER, file_name)

        ok = download_file(url, save_base)
        results.append(1 if ok else 0)

    if "status" in df.columns:
        df["status"] = results
    else:
        df.insert(0, "status", results)

    output_file = file_path.replace(".xlsx", "_sonuc.xlsx")
    df.to_excel(output_file, index=False)

    print("\n✅ Bitti:", output_file)


# 🔥 MENÜ SİSTEMİ
def menu():
    while True:
        print("\n==============================")
        print("  EXCEL DOWNLOADER SYSTEM")
        print("==============================")
        print("1) Excel dosyası seç (drag & drop)")
        print("2) Manuel dosya yolu gir")
        print("3) Çıkış")
        print("==============================")

        choice = input("Seçim (1-3): ").strip()

        # 1 - drag & drop
        if choice == "1":
            path = input("Excel dosyasını buraya sürükle bırak: ").strip().strip('"')
            # if os.path.exists(path):
            path = normalize_path(path)
            if os.path.exists(path):
                
                process_excel(path)
            else:
                print("❌ Dosya bulunamadı!")

        # 2 - manual
        elif choice == "2":
            path = input("Dosya yolu gir: ").strip().strip('"')
            if os.path.exists(path):
                process_excel(path)
            else:
                print("❌ Dosya bulunamadı!")

        # 3 - exit
        elif choice == "3":
            print("👋 Çıkılıyor...")
            break

        else:
            print("❌ Geçersiz seçim!")


if __name__ == "__main__":
    # Eğer direkt drag & drop yapılırsa
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        process_excel(file_path)
    else:
        menu()