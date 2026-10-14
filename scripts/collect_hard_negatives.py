import os
import sys
import io
import time
import json
import random
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train" / "bukan_banjir"
VAL_DIR = BASE_DIR / "dataset" / "val" / "bukan_banjir"

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENT = "FloodClassifierResearch/1.0 (academic research; contact: student@univ.ac.id)"

EXCLUDED_KEYWORDS = [
    "flood", "flooding", "flooded", "inundat", "inundation", "deluge", 
    "disaster", "overflow", "overflowing", "submerged", "submerge", "tsunami", "drown"
]

SEARCH_QUERIES = [
    "wet asphalt road",
    "wet road rain reflection",
    "puddle on asphalt road",
    "rainy street city car",
    "city road traffic asphalt",
    "calm river water bridge",
    "normal canal water urban",
    "dry street suburban road",
    "urban pedestrian walkway street"
]

def search_wikimedia_images(query, limit=25):
    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&list=search"
        f"&srsearch={encoded_query}&srnamespace=6&srlimit={limit}&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
            results = data.get("query", {}).get("search", [])
            titles = [item["title"] for item in results]
            return titles
    except Exception as e:
        print(f"[WARN] Error searching query '{query}': {e}")
        return []

def get_image_direct_url(title):
    encoded_title = urllib.parse.quote(title)
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&titles={encoded_title}"
        f"&prop=imageinfo&iiprop=url&format=json"
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as response:
            data = json.loads(response.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                info = page.get("imageinfo", [])
                if info and "url" in info[0]:
                    img_url = info[0]["url"]
                    clean_url = img_url.split("?")[0]
                    if clean_url.lower().endswith((".jpg", ".jpeg", ".png")):
                        return img_url
    except Exception as e:
        pass
    return None

def download_and_process_image(img_url, max_size=640):
    req = urllib.request.Request(img_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw_bytes = response.read()
            if len(raw_bytes) < 10240: # skip if < 10KB
                return None
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            
            # Resize agar hemat disk dan cepat
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            return img
    except Exception as e:
        return None

def run_collection(target_count=180):
    print("=" * 60)
    print("MENGUMPULKAN CITRA HARD NEGATIVES (BUKAN BANJIR)")
    print("=" * 60)
    print(f"Target gambar: {target_count}")
    
    seen_titles = set()
    downloaded = 0
    train_count = 0
    val_count = 0

    for query in SEARCH_QUERIES:
        if downloaded >= target_count:
            break
        print(f"\n[QUERY] Mencari: '{query}'...")
        titles = search_wikimedia_images(query, limit=35)
        print(f"        Ditemukan {len(titles)} berkas.")
        
        for title in titles:
            if downloaded >= target_count:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Cek apakah judul mengandung kata kunci banjir
            title_lower = title.lower()
            if any(k in title_lower for k in EXCLUDED_KEYWORDS):
                continue
            
            img_url = get_image_direct_url(title)
            if not img_url:
                continue
                
            img = download_and_process_image(img_url)
            if img is None:
                continue
                
            downloaded += 1
            # 80% train, 20% val
            is_val = (random.random() < 0.20)
            target_dir = VAL_DIR if is_val else TRAIN_DIR
            filename = f"hardneg_{int(time.time())}_{downloaded:04d}.jpg"
            target_path = target_dir / filename
            
            img.save(target_path, "JPEG", quality=88)
            
            if is_val:
                val_count += 1
            else:
                train_count += 1
                
            if downloaded % 5 == 0 or downloaded == target_count:
                print(f"  [PROGRES] Total: {downloaded}/{target_count} (Train: {train_count}, Val: {val_count})")
            
            time.sleep(0.15) # Hormati rate limit server

    print("\n" + "=" * 60)
    print(f"SELESAI! Berhasil mengumpulkan {downloaded} gambar baru.")
    print(f"-> Train bukan_banjir : +{train_count} citra")
    print(f"-> Val bukan_banjir   : +{val_count} citra")
    print("=" * 60)

if __name__ == "__main__":
    count_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    run_collection(count_arg)
