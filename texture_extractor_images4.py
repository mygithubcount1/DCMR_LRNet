import os
import cv2
import numpy as np
from PIL import Image
from skimage import filters, morphology, measure
from skimage.feature import local_binary_pattern
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- 1. Generate Solid Mask ----------
def extract_borders(image_path, output_dir):
    """Return 0/255 solid mask; save original image at the same time"""
    os.makedirs(output_dir, exist_ok=True)

    # Save original image
    Image.open(image_path).save(os.path.join(output_dir, os.path.basename(image_path)))

    img = np.array(Image.open(image_path).convert('L'))
    thr = filters.threshold_otsu(img)
    bin_img = (img > thr).astype(np.uint8) * 255
    bin_img = morphology.opening(bin_img, morphology.disk(3))
    bin_img = morphology.closing(bin_img, morphology.disk(3))
    bin_img = morphology.remove_small_objects(bin_img.astype(bool), 100, connectivity=1)

    label_img = measure.label(bin_img, connectivity=1)
    regions = measure.regionprops(label_img)
    if not regions:
        return np.zeros_like(bin_img, dtype=np.uint8)

    largest = max(regions, key=lambda r: r.area)
    mask = np.zeros_like(bin_img, dtype=np.uint8)
    mask[largest.bbox[0]:largest.bbox[2],
         largest.bbox[1]:largest.bbox[3]] = largest.image.astype(np.uint8) * 255
    return mask

# ---------- 2. Fill Background with White + LBP ----------
def extract_veins(image_path, mask):
    """Return: grayscale image with white background, LBP image"""
    gray = cv2.imread(image_path, 0)

    # Fill area outside contour with white
    mask = (mask == 255)           # True means outside contour
    gray[mask] = 255                   # Fill with white

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius)
    lbp = np.uint8((lbp / lbp.max()) * 255)

    # lbp[mask == 255] = 255          # Fill outside contour with white

    return gray, lbp

# ---------- 3. Process Single Image ----------
def process_one_file(src_path, dst_root):
    """Process a single file, keep directory structure"""
    try:
        # Calculate output subdirectory
        rel_dir = os.path.relpath(os.path.dirname(src_path), input_root)
        dst_dir = os.path.join(dst_root, rel_dir)
        os.makedirs(dst_dir, exist_ok=True)

        print(f"-> {os.path.relpath(src_path, input_root)}")

        mask = extract_borders(src_path, dst_dir)
        masked_gray, lbp_img = extract_veins(src_path, mask)

        base = os.path.splitext(os.path.basename(src_path))[0]
        Image.fromarray(lbp_img).save(os.path.join(dst_dir, f"vein2_{base}.png"))

    except Exception as e:
        print(f"ERROR {src_path}: {e}")

# ---------- 4. Recursive Scan + Thread Pool ----------
def collect_files(root):
    """Recursively collect all .jpg/.png files"""
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.png')):
                yield os.path.join(dirpath, f)

def run(root_in, root_out):
    files = list(collect_files(root_in))
    print(f"Total {len(files)} images, start processing...")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = [pool.submit(process_one_file, f, root_out) for f in files]
        for _ in as_completed(futures):
            pass  # Exceptions are caught in process_one_file

# ---------- 5. Main Entry ----------
if __name__ == '__main__':
    input_root = r'images'      # Original multi-level directory
    output_root = r'images4'    # Output keeps the same structure
    run(input_root, output_root)