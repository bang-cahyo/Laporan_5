# ======================================
# utils.py
# ======================================
import numpy as np
from PIL import Image
import io
import cv2

# ======================================
# Fungsi untuk mengubah numpy array menjadi image yang bisa di-download
# ======================================
def get_downloadable_image(np_img):
    """Convert numpy array to downloadable PNG bytes."""
    image = Image.fromarray(np_img)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

# ======================================
# Fungsi Letterbox Resize untuk YOLO
# ======================================
def letterbox_image(img, target_size=(640, 640)):
    """
    Resize image keeping aspect ratio with padding.
    img: numpy array RGB
    target_size: (width, height)
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    # Resize image
    img_resized = cv2.resize(img, (nw, nh))
    
    # Buat canvas berwarna abu-abu
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    
    # Hitung padding
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    
    # Letakkan gambar yang sudah di-resize ke canvas
    canvas[top:top+nh, left:left+nw, :] = img_resized
    
    return canvas
