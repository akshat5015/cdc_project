import os
import pandas as pd
import contextily as cx
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch

# --- CONFIG ---
SAVE_DIR = r"C:\Users\aksha\OneDrive\Desktop\satellite\property_images"
ZOOM = 19  # Pushing to the highest common resolution
THREADS = 15 

# Verify GPU for later steps
if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️ No GPU found, using CPU.")
    device = "cpu"

def download_image_house_focus(row):
    lat, lon, prop_id = row['lat'], row['long'], row['id']
    save_path = os.path.join(SAVE_DIR, f"{prop_id}.png")
    
    if os.path.exists(save_path): return True

    try:
        # Tighter margin (approx 40m x 40m) to focus ONLY on the property
        margin = 0.00025 
        west, south, east, north = (lon - margin, lat - margin, lon + margin, lat + margin)
        
        # Fetching at Zoom 19
        img, extent = cx.bounds2img(west, south, east, north, zoom=ZOOM, 
                                    source=cx.providers.Esri.WorldImagery, ll=True)
        
        Image.fromarray(img).convert("RGB").save(save_path)
        return True
    except Exception:
        # Fallback to Zoom 18 if 19 isn't available
        try:
            img, extent = cx.bounds2img(west, south, east, north, zoom=18, 
                                        source=cx.providers.Esri.WorldImagery, ll=True)
            Image.fromarray(img).convert("RGB").save(save_path)
            return True
        except:
            return False

# --- EXECUTION ---
df = pd.read_csv(r"C:\Users\aksha\OneDrive\Desktop\satellite\data\train(1)(train(1)).csv")
rows = [row for _, row in df.iterrows()]

print(f"Starting High-Res Download (Zoom {ZOOM})...")
with ThreadPoolExecutor(max_workers=THREADS) as executor:
    list(tqdm(executor.map(download_image_house_focus, rows), total=len(df), desc="Fetching House Detail"))