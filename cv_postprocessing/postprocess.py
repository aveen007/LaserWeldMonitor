import numpy as np
import os
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
# from pixel_size.utils import get_pixel_real_size
from weld_processing.read_mask import plot_mask_and_point,return_points_and_size
from PIL import Image, ImageDraw
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def overlay_mask(image_path, mask, alpha=0.8):
    # Load the image
    # image = Image.open(image_path).convert("RGB")
    image = Image.open(image_path).convert("RGBA")
    # resized_image = image.resize(384,480)

    # Load the mask and convert it to an array
    mask = Image.open(mask_path).convert("L")  # Convert to grayscale
    mask = np.array(mask)

    # Create a colored version of the mask
    colored_mask = np.zeros((*mask.shape, 4), dtype=np.uint8)  # RGBA
    colored_mask[:, :, 0] = 255  # Red channel (color of the mask)
    colored_mask[:, :, 3] = mask * 255  # Alpha channel based on the mask

    # Convert the colored mask to an image
    colored_mask_image = Image.fromarray(colored_mask, 'RGBA')

    # Overlay the mask on the image with specified transparency
    overlay = Image.alpha_composite(image, colored_mask_image)

    # Save the result to the specified path
    overlay_path = "./predicted_masks/overlay/" + image_path.split('/')[3]
    overlay.save(overlay_path)
    
    # Convert overlay to format compatible with OpenCV (BGR) for further processing (if necessary)
    overlay_cv = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGR)
    cv2.imwrite(overlay_path, overlay_cv)

# Example usage:
# overlay_mask('./predicted_masks/images/00435-4194690285.png', './predicted_masks/masks/00435-4194690285.png')

def overlay_mask_jpg(image_path, masks, alpha=0.1):
    """
    Overlays a YOLO segmentation mask onto an image, ensuring only one connected polygon is drawn.
    
    Args:
        image_path (str): Path to the input image.
        masks (list): List of YOLO masks (polygon coordinates).
        alpha (float): Transparency level for the overlay.
    
    Saves:
        Processed image with overlay mask.
    """
    overlay_path = "./predicted_masks/overlay/" + image_path.split('/')[-1]  # Use last part of path
    
    # Open image in PIL
    img_pil = Image.open(image_path).convert("RGBA")
    overlay = Image.new('RGBA', img_pil.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    all_polygons = []

    # Collect all polygons from masks
    for mask in masks:
        polygon = mask.xy[0]  # Get polygon coordinates
        if len(polygon) >= 3:
            all_polygons.append(Polygon(polygon))  # Convert to Shapely polygon

    if not all_polygons:
        print("No valid polygons detected.")
        return  # Exit if no valid polygons

    # Merge polygons and keep only the largest one
    merged = MultiPolygon(all_polygons).buffer(0)  # Merge overlapping polygons
    if isinstance(merged, MultiPolygon):
        largest_polygon = max(merged.geoms, key=lambda p: p.area)  # Keep the largest polygon
    else:
        largest_polygon = merged

    # Convert the largest polygon back to a format usable by ImageDraw
    polygon_coords = list(largest_polygon.exterior.coords)

    # Draw the largest polygon
    overlay_draw.polygon(polygon_coords, fill=(0, 255, 0, 50))

    # Draw a centroid marker
    centroid = largest_polygon.centroid
    circle_radius = 5
    left_up_point = (centroid.x - circle_radius, centroid.y - circle_radius)
    right_down_point = (centroid.x + circle_radius, centroid.y + circle_radius)
    overlay_draw.ellipse([left_up_point, right_down_point], fill=(255, 0, 0))

    # Merge overlay with original image
    img_pil = Image.alpha_composite(img_pil, overlay)

    # Convert PIL image to OpenCV format and save
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, frame)

    print(f"Overlay saved: {overlay_path}")
def overlay_mask_unified(image_path, mask_input, alpha=0.3):
    import cv2
    import numpy as np
    from pathlib import Path
    from shapely.geometry import Polygon

    image_path = Path(image_path)
    overlay_base = Path("./predicted_masks/overlay")
    overlay_path = overlay_base / image_path.relative_to("./predicted_masks/images")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not read image {image_path}")
        return
    overlay = np.zeros_like(img, dtype=np.uint8)

    polygons = []

    # ---- Handle YOLO Masks object ----
    # YOLO Masks object has attribute 'xy'
    if hasattr(mask_input, 'xy'):
        for poly in mask_input.xy:
            polygons.append(np.array(poly, dtype=np.int32))
    
    # ---- Handle list of polygons ----
    elif isinstance(mask_input, list):
        for poly in mask_input:
            polygons.append(np.array(poly, dtype=np.int32))
    
    # ---- Handle raster mask path ----
    else:
        mask_path = Path(mask_input)
        if mask_path.exists():
            import PIL.Image as Image
            mask_img = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask_img)
            mask_bin = (mask_arr > 128).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [cnt.reshape(-1,2) for cnt in contours]

    # Draw polygons
    for poly in polygons:
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [poly], (0, 255, 0))
            c = np.mean(poly, axis=0).astype(int)
            cv2.circle(overlay, tuple(c), 4, (0,0,255), -1)

    blended = cv2.addWeighted(img, 1, overlay, alpha, 0)
    cv2.imwrite(str(overlay_path), blended)
    print(f"[OK] Overlay saved: {overlay_path}")
    
def process_yolo_masks(masks):
    """
    Takes YOLO segmentation masks and returns a single connected binary mask.
    
    Args:
        masks (tensor): YOLOv8 segmentation mask tensor.
        
    Returns:
        final_mask (numpy.ndarray): Processed binary mask with only one connected component.
    """
    if masks is None or len(masks) == 0:
        return None  # No mask detected

    # Convert YOLO mask tensor to NumPy
    masks = masks.data.cpu().numpy()  # Shape: (num_masks, H, W)

    # Merge all masks into a single binary mask
    merged_mask = np.max(masks, axis=0)  # Combine multiple masks into one

    # Convert to uint8 format for OpenCV
    merged_mask = (merged_mask * 255).astype(np.uint8)

    # Apply morphological closing to fill small gaps
    kernel = np.ones((5, 5), np.uint8)
    merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, kernel)

    # Find the largest connected component
    contours, _ = cv2.findContours(merged_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        final_mask = np.zeros_like(merged_mask)
        cv2.drawContours(final_mask, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    else:
        final_mask = merged_mask  # If no contours found, keep original

    return final_mask
def overlay_mask_multiclass(image_path, results, alpha=0.3):
    import cv2
    import numpy as np
    from pathlib import Path
    
    # Define class colors - BGR format for OpenCV
    class_colors = {
        0: (255, 0, 0),      # blue_scale - Blue
        1: (0, 255, 255),    # box scale - Yellow
        2: (128, 128, 128),  # plate - Grayish
        3: (255, 255, 0),    # ruler - Cyan
        4: (0, 255, 0),      # scale_bar - Green
        5: (0, 0, 255)       # weld - Reddish
    }
    
    class_names = ['blue_scale', 'box scale', 'plate', 'ruler', 'scale_bar', 'weld']
    
    image_path = Path(image_path)
    overlay_base = Path("./predicted_masks/overlay")
    overlay_path = overlay_base / image_path.relative_to("./predicted_masks/images")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Could not read image {image_path}")
        return
    
    overlay = np.zeros_like(img, dtype=np.uint8)
    
    # Check if we have detection results
    if results is None or len(results) == 0 or results[0].masks is None:
        print(f"[INFO] No detections for {image_path}")
        # Save original image if no detections
        cv2.imwrite(str(overlay_path), img)
        return

    result = results[0]
    
    # Process each detection
    for i, (mask, class_id) in enumerate(zip(result.masks.xy, result.boxes.cls)):
        class_id = int(class_id)
        color = class_colors.get(class_id, (0, 255, 0))  # Default to green
        
        polygon = np.array(mask, dtype=np.int32)
        
        if len(polygon) >= 3:
            # Draw filled polygon
            cv2.fillPoly(overlay, [polygon], color)
            
            # Draw centroid
            centroid = np.mean(polygon, axis=0).astype(int)
            cv2.circle(overlay, tuple(centroid), 4, (255, 255, 255), -1)  # White centroid
            
            # Optional: Add class label
            label = f"{class_names[class_id]}"
            cv2.putText(overlay, label, tuple(centroid - [20, 10]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    blended = cv2.addWeighted(img, 1, overlay, alpha, 0)
    cv2.imwrite(str(overlay_path), blended)
    print(f"[OK] Multi-class overlay saved: {overlay_path}")
def normalize_illumination(img: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to normalize illumination."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.equalizeHist(gray)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
def save_prediction_masks(model_path_str):
    PATH_Images = Path("./predicted_masks/images")
    PATH_Masks = Path("./predicted_masks/masks")
    
    model_path = Path(model_path_str)
    model = YOLO(model_path)

    # All supported image extensions
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.JPG', '.PNG', '.TIF'}

    # Walk recursively through subfolders
    image_files = [p for p in PATH_Images.rglob('*') if p.suffix in valid_exts]

    for img_path in tqdm(image_files, desc="Processing images"):
        rel_path = img_path.relative_to(PATH_Images)
        mask_output_path = PATH_Masks / rel_path
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        H, W, _ = img.shape


        results = model(img, imgsz=[640], iou=0.4, conf=0.1)

        if results[0].masks is None or len(results[0].masks) == 0:
            # 🔹 2nd pass — retry with illumination normalization
            img_norm = normalize_illumination(img)
            results = model(img_norm, imgsz=[640], iou=0.4, conf=0.1)

            if results[0].masks is None or len(results[0].masks) == 0:
                print(f"[WARN] No masks predicted for {img_path} (even after normalization)")
                mask = np.zeros((H, W), dtype=np.uint8)
                cv2.imwrite(str(mask_output_path), mask)
                overlay_mask_unified(str(img_path), [])
                continue
            else:
                print(f"[INFO] {img_path.name}: Detected only after illumination normalization")

        # 🔹 Process mask (successful detection)
        mask = process_yolo_masks(results[0].masks)
        mask = cv2.resize(mask, (W, H))
        cv2.imwrite(str(mask_output_path), mask)
        # overlay_mask_unified(str(img_path), results[0].masks)
        overlay_mask_multiclass(str(img_path), results)
def generate_segmentation_excel(base_dir='./predicted_masks/images', output_excel='segmentation_report.xlsx'):
    base = Path(base_dir)
    valid_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.JPG', '.PNG', '.TIF'}

    data_by_material = {}

    # Walk through all files recursively
    for img_path in base.rglob('*'):
        if img_path.suffix not in valid_exts:
            continue

        # Extract folder hierarchy
        rel_parts = img_path.relative_to(base).parts
        if len(rel_parts) < 2:
            continue  # expects at least material/image.jpg

        material = rel_parts[0]
        subfolders = rel_parts[1:-1] if len(rel_parts) > 2 else []
        filename = rel_parts[-1]

        row = {
            'image_id': filename,
            'relative_path': str(img_path.relative_to(base)),
            'material': material,
            'subfolder_path': '/'.join(subfolders) if subfolders else '',
            'segmentation_quality': '',   # to fill manually
            'failure_type': '',
            'notes': '',
            'include_in_train': '',
            'include_in_test': ''
        }

        data_by_material.setdefault(material, []).append(row)

    # Write each material as a separate sheet
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        for material, rows in data_by_material.items():
            df = pd.DataFrame(rows)
            df.to_excel(writer, sheet_name=material[:31], index=False)

    print(f"✅ Excel report saved to {output_excel}")
import shutil
def add_to_dataset():
    path_images = './tmp/images/'
    path_masks= './tmp/masks/'
    path_masks_predicted='./predicted_masks/masks/'
    path_images_predicted='./predicted_masks/images/'
    path_overlay='./predicted_masks/overlay/'
    list_img=[img for img in os.listdir(path_overlay) if img.endswith('.JPG')==True]
    for i in tqdm(list_img):
        img_path_predicted=path_images_predicted+"/"+ i
        mask_path_predicted=path_masks_predicted+"/"+ i
          # Move predicted image to the image directory
        if os.path.exists(img_path_predicted):
            shutil.move(img_path_predicted, path_images)
        
        # Move predicted mask to the masks directory
        if os.path.exists(mask_path_predicted):
            shutil.move(mask_path_predicted, path_masks)