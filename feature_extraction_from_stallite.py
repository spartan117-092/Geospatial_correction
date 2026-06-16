import os
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, box
from rasterio.features import rasterize

# Scikit-Image Dependencies for Border Extraction
from skimage import color, filters, feature
from skimage.morphology import remove_small_objects, binary_dilation, disk
from skimage.measure import label, regionprops

def extract_and_save_fixed_patch(geojson_path, plot_no, tiff_imagery="imagery.tif", output_dir="dataset_patches", target_size=500, sigma=3, low_threshold=0.1, high_threshold=0.25, min_length=50):
    """
    Extracts a fixed 500x500 pixel imagery segment from a raw satellite geotiff,
    runs a localized Canny edge engine to isolate high-confidence field boundaries,
    and saves both arrays as NumPy matrices for ML datasets.
    """
    # Create export directory structure
    images_dir = os.path.join(output_dir, "images")
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # 1. Load Vector Dataset and Isolate the Specific Plot Center
    gdf = gpd.read_file(geojson_path)
    if plot_no >= len(gdf):
        print(f"Error: Plot number {plot_no} is out of bounds.")
        return False
        
    target_row = gdf.iloc[[plot_no]].copy()
    raw_geom = target_row.geometry.values[0]
    
    if isinstance(raw_geom, MultiPolygon):
        vector_poly = max(raw_geom.geoms, key=lambda p: p.area)
    else:
        vector_poly = raw_geom
        
    # 2. Open Raster to Calculate Spatial Bounds for a 500x500 Window
    with rasterio.open(tiff_imagery) as src_img:
        if target_row.crs != src_img.crs:
            target_row = target_row.to_crs(src_img.crs)
            raw_geom = target_row.geometry.values[0]
            vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
            
        centroid = vector_poly.centroid
        center_x, center_y = centroid.x, centroid.y
        
        pixel_width = abs(src_img.transform.a)
        pixel_height = abs(src_img.transform.e)
        
        half_width_meters = (target_size / 2) * pixel_width
        half_height_meters = (target_size / 2) * pixel_height
        
        bbox = box(
            center_x - half_width_meters,
            center_y - half_height_meters,
            center_x + half_width_meters,
            center_y + half_height_meters
        )
        
        cropped_raster, _ = rasterio.mask.mask(src_img, [bbox], crop=True)
        
    # 3. Format Channel Layout (From Raster Shape to Standard Image Shape)
    if cropped_raster.shape[0] == 3:    # RGB Layout
        img_patch = np.moveaxis(cropped_raster, 0, -1)
    elif cropped_raster.shape[0] == 4:  # RGBA Layout
        img_patch = np.moveaxis(cropped_raster[:3], 0, -1)
    else:                               # Grayscale Single-Band
        img_patch = cropped_raster[0]
        
    # Force alignment to absolute dimensions to avoid rounding variations
    if img_patch.shape[0] != target_size or img_patch.shape[1] != target_size:
        if len(img_patch.shape) == 3:
            img_patch = img_patch[:target_size, :target_size, :]
        else:
            img_patch = img_patch[:target_size, :target_size]
        
    # Normalize pixel data cleanly to standard 0-255 uint8 range
    if img_patch.max() > 255:
        img_patch = ((img_patch - img_patch.min()) / (img_patch.max() - img_patch.min()) * 255).astype(np.uint8)
    else:
        img_patch = img_patch.astype(np.uint8)

    # ------------------------------THIS IS THE NEIGHBOURHOOD DIRECTIONAL PCA SAME I USED IN THE BOUNDARY ALIGNMENT------------------------------------
    

        
    # =========================================================================
    # FIXED: Custom Matrix Gradient Discriminator & Non-Linear Purge
    # =========================================================================
    # 1. Force the image patch to grayscale
    if len(img_patch.shape) == 3:
        gray_patch = (img_patch[:, :, 0] * 0.299 + 
                      img_patch[:, :, 1] * 0.587 + 
                      img_patch[:, :, 2] * 0.114).astype(float)
    else:
        gray_patch = img_patch.astype(float)

    # 2. Compute sudden pixel changes using raw matrix shifts (Sobel style)
    # This finds sharp transitions in X and Y directions instantly
    grad_x = np.zeros_like(gray_patch)
    grad_y = np.zeros_like(gray_patch)
    
    grad_x[:, 1:-1] = gray_patch[:, 2:] - gray_patch[:, :-2]
    grad_y[1:-1, :] = gray_patch[2:, :] - gray_patch[:-2, :]
    
    # Calculate total gradient magnitude at every coordinate
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient values to a clean 0.0 to 1.0 window
    if gradient_magnitude.max() > 0:
        gradient_magnitude /= gradient_magnitude.max()

    # 3. Apply sudden change threshold
    # Pixels that change sharper than 18% of the local max are registered as features
    feature_mask = gradient_magnitude > 0.18

    # 3. Apply sudden change threshold to create raw binary mask
    raw_feature_mask = gradient_magnitude > 0.18
    
    # Get coordinates of all white pixels that crossed the gradient threshold
    rows, cols = np.where(raw_feature_mask)
    white_pts = np.column_stack((cols, rows))  # shape: (N, 2)
    
    # Initialize your clean, high-confidence mask
    mask_patch = np.zeros_like(raw_feature_mask, dtype=bool)
    
    # Local Circle Scan Configuration
    local_radius_px = 15.0
    
    # 4. FIXED: Local PCA Direction Check to Purge Blobs and Random Meshes
    for pt in white_pts:
        # Calculate distance from this pixel to all other white pixels
        deltas = white_pts - pt
        distances = np.linalg.norm(deltas, axis=1)
        
        # Isolate the neighborhood circle points
        neighborhood = white_pts[distances <= local_radius_px]
        
        # Rule Check: A valid line segment must have a minimum point density
        if len(neighborhood) < 5:
            continue
            
        # Compute the local geometric distribution matrix
        centroid = np.mean(neighborhood, axis=0)
        centered_matrix = neighborhood - centroid
        covariance_matrix = np.cov(centered_matrix, rowvar=False)
        
        if covariance_matrix.shape != (2, 2) or np.any(np.isnan(covariance_matrix)):
            continue
            
        # Extract structural eigenvalues
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        lam_min, lam_max = eigenvalues[0], eigenvalues[1]
        
        # Inside your loop, right after calculating eigenvalues:
        if lam_max > 0:
            linear_ratio = lam_min / lam_max
            
            # --- THE TWO-LAYER VALIDATION GATE ---
            # 1. Linearity Rule: Must be clean and directional
            is_linear = linear_ratio <= 0.15
            
            # 2. Density Rule: A true line passing through a 5px radius circle 
            # can contain at most 12-15 pixels. If it contains 25+, it's a massive blob!
            is_not_a_blob = len(neighborhood) <= 14
            
            if is_linear and is_not_a_blob:
                mask_patch[pt[1], pt[0]] = True

    # Thickening for training label visualization stability
    mask_patch = binary_dilation(mask_patch, disk(1))
    mask_patch_uint8 = (mask_patch * 255).astype(np.uint8)





    # 4. Filter out non-linear crop row clusters using regional properties
    labels = label(feature_mask)
    mask_patch = np.zeros_like(feature_mask, dtype=bool)

    for region in regionprops(labels):
        # Rule check: Neglect small noise dots
        if region.area < min_length:
            continue
            
        # Linearity check using Eccentricity (0 = perfect circle, 1 = perfect straight line)
        # Structural field boundaries are long and straight, so their eccentricity is very high (> 0.92)
        # Crop rows or patches of trees form chunky blocks with low eccentricity
        if region.eccentricity >= 0.9:
            mask_patch[labels == region.label] = True

    # Thickening for training label visualization stability
    mask_patch = binary_dilation(mask_patch, disk(1))
    mask_patch_uint8 = (mask_patch * 255).astype(np.uint8)
    # =========================================================================

    # 5. Export Pristine NumPy Matrices to Disk
    img_filename = f"plot_{plot_no}_img.npy"
    mask_filename = f"plot_{plot_no}_mask.npy"
    
    np.save(os.path.join(images_dir, img_filename), img_patch)
    np.save(os.path.join(masks_dir, mask_filename), mask_patch_uint8)
    
    # 6. Quality Control Verification Display Overlay (Borders marked in Red)
    overlay_preview = img_patch.copy()
    if len(overlay_preview.shape) == 3:
        overlay_preview[mask_patch] = [255, 0, 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    if len(img_patch.shape) == 3:
        ax1.imshow(img_patch)
    else:
        ax1.imshow(img_patch, cmap='gray')
    ax1.set_title(f"Original Sat Image Patch ({target_size}x{target_size})", fontsize=10, fontweight='bold')
    ax1.axis('off')
    
    if len(img_patch.shape) == 3:
        ax2.imshow(overlay_preview)
    else:
        ax2.imshow(mask_patch_uint8, cmap='gray')
    ax2.set_title("Extracted High Confidence Target Borders", fontsize=10, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    preview_path = os.path.join(output_dir, f"plot_{plot_no}_verification_{plot_no}.png")
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close() # Keep RAM clear during automated loop pipelines
    
    print(f" Success: Exported dataset elements for Plot {plot_no}:")
    print(f" -> Sat Image Patch : {os.path.join(images_dir, img_filename)}")
    print(f" -> Ground-Truth Mask: {os.path.join(masks_dir, mask_filename)}")
    
    return True


# plot_no = 1
# extract_and_save_fixed_patch("input.geojson",plot_no)


# import numpy as np
# import matplotlib.pyplot as plt

# # Load the saved target mask from your directory
# # (Make sure to point this to your actual file path)
# mask_path = f"dataset_patches/masks/plot_{plot_no}_mask.npy"
# mask_array = np.load(mask_path)

# # Display the pure black and white boundary map
# plt.figure(figsize=(8, 8))
# plt.imshow(mask_array, cmap='gray')
# plt.title("Pure Black & White Target Mask", fontsize=12, fontweight='bold')
# plt.axis("off")  # Hides the pixel coordinate axes

# plt.show()



import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt

# =========================================================================
# 1. THE DIAGNOSTIC VISUALIZER ENGINE
# =========================================================================
def diagnose_local_pca_behavior(white_pts, candidate_pt, local_radius_px=5.0):
    """
    Plots the local neighborhood structure around a pixel along with its 
    computed PCA variance axes to verify structural linearity.
    """
    deltas = white_pts - candidate_pt
    distances = np.linalg.norm(deltas, axis=1)
    neighborhood = white_pts[distances <= local_radius_px]
    
    if len(neighborhood) < 2:
        return None
        
    centroid = np.mean(neighborhood, axis=0)
    centered_matrix = neighborhood - centroid
    covariance_matrix = np.cov(centered_matrix, rowvar=False)
    
    if covariance_matrix.ndim < 2:
        return None
        
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(neighborhood[:, 0], neighborhood[:, 1], color='blue', s=40, label=f'Neighbors ({len(neighborhood)} pts)')
    ax.scatter(candidate_pt[0], candidate_pt[1], color='red', s=120, marker='*', zorder=5, label='Target Pixel')
    
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        v_length = 2.0 * np.sqrt(max(0, lam))
        end_pt = centroid + vec * v_length
        color = 'green' if i == 1 else 'magenta'
        label_str = f'Max Axis (λ_max={lam:.2f})' if i == 1 else f'Min Axis (λ_min={lam:.2f})'
        ax.plot([centroid[0], end_pt[0]], [centroid[1], end_pt[1]], color=color, linewidth=3, label=label_str)
        
    circle = plt.Circle((candidate_pt[0], candidate_pt[1]), local_radius_px, color='black', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 1.0
    ax.set_title(f"Local PCA Neighborhood\nRatio (λ_min/λ_max): {ratio:.3f} | Density: {len(neighborhood)}")
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.show()

# =========================================================================
# 2. THE MAIN INDEPENDENT FEATURE EXTRACTION & DEBUG ENGINE
# =========================================================================
def debug_single_plot_features(geojson_path, plot_no, tiff_imagery="imagery.tif", target_size=500):
    """
    Independent debugging pipeline that loads a raster segment, computes
    the gradient matrix, filters out jagged mesh noise using custom geometric 
    gates, and runs interactive node-level PCA diagnostics.
    """
    # Step 1: Load Vector Dataset and Isolate the Target Plot
    gdf = gpd.read_file(geojson_path)
    if plot_no >= len(gdf):
        print(f"Error: Plot number {plot_no} is out of bounds.")
        return False
        
    target_row = gdf.iloc[[plot_no]].copy()
    raw_geom = target_row.geometry.values[0]
    vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
    
    # Step 2: Open Raster to Extract the 500x500 Coordinate Envelope
    with rasterio.open(tiff_imagery) as src_img:
        if target_row.crs != src_img.crs:
            target_row = target_row.to_crs(src_img.crs)
            raw_geom = target_row.geometry.values[0]
            vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
            
        centroid = vector_poly.centroid
        half_width_meters = (target_size / 2) * abs(src_img.transform.a)
        half_height_meters = (target_size / 2) * abs(src_img.transform.e)
        
        bbox = box(
            centroid.x - half_width_meters, centroid.y - half_height_meters,
            centroid.x + half_width_meters, centroid.y + half_height_meters
        )
        cropped_raster, _ = rasterio.mask.mask(src_img, [bbox], crop=True)
        
    # Step 3: Format Channel Layout and Force Absolute Dimension Matching
    if cropped_raster.shape[0] >= 3:
        img_patch = np.moveaxis(cropped_raster[:3], 0, -1)
        gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114).astype(float)
    else:
        gray_patch = cropped_raster[0].astype(float)
        
    gray_patch = gray_patch[:target_size, :target_size]

    # Step 4: Compute Sudden Pixel Changes Using Element-Wise Matrix Shifts
    grad_x = np.zeros_like(gray_patch)
    grad_y = np.zeros_like(gray_patch)
    grad_x[:, 1:-1] = gray_patch[:, 2:] - gray_patch[:, :-2]
    grad_y[1:-1, :] = gray_patch[2:, :] - gray_patch[:-2, :]
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if gradient_magnitude.max() > 0:
        gradient_magnitude /= gradient_magnitude.max()

    # Step 5: Threshold to Generate the Binary Candidate Mask Matrix
    raw_feature_mask = gradient_magnitude > 0.18
    
    rows, cols = np.where(raw_feature_mask)
    white_pts = np.column_stack((cols, rows))
    
    # Initialize our clean, high-confidence target array
    mask_patch = np.zeros_like(raw_feature_mask, dtype=bool)
    local_radius_px = 5.0
    
    # =========================================================================
    # THE STRUCTURAL PURGE: ADVANCED NOISE MESH FILTERS
    # =========================================================================
    for pt in white_pts:
        deltas = white_pts - pt
        distances = np.linalg.norm(deltas, axis=1)
        neighborhood = white_pts[distances <= local_radius_px]
        
        # Rule 1: Point Density Filter (Chokes out empty zones and oversized blocks)
        if len(neighborhood) < 5 or len(neighborhood) > 15:
            continue
            
        centroid = np.mean(neighborhood, axis=0)
        
        # UPGRADE 1: THE CENTRALITY GATE
        # True lines pass right through the center. Off-center pixels on the jagged edges 
        # of tree canopies or crop row fields get dropped instantly.
        dist_to_centroid = np.linalg.norm(pt - centroid)
        if dist_to_centroid > 1.8:  
            continue
            
        covariance_matrix = np.cov(neighborhood - centroid, rowvar=False)
        
        if covariance_matrix.shape != (2, 2) or np.any(np.isnan(covariance_matrix)):
            continue
            
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        lam_min, lam_max = eigenvalues[0], eigenvalues[1]
        
        if lam_max > 0:
            linear_ratio = lam_min / lam_max
            
            # UPGRADE 2 & 3: STRUCTURAL DIMENSION GATES
            is_perfectly_linear = linear_ratio <= 0.12 # Tightened line threshold
            has_structural_length = lam_max >= 1.5     # Rejects short random fragments
            
            if is_perfectly_linear and has_structural_length:
                mask_patch[pt[1], pt[0]] = True
    # =========================================================================

    print(f"\n--- Running Diagnostics for Plot {plot_no} ---")
    print(f"Total candidate points found in gradient mask: {len(white_pts)}")
    
    # Re-gather only the remaining high-confidence coordinates for visual tracking
    final_rows, final_cols = np.where(mask_patch)
    filtered_white_pts = np.column_stack((final_cols, final_rows))
    
    if len(filtered_white_pts) == 0:
        print("Notice: No feature points survived the geometric gates.")
        return False

    # Sample up to 3 coordinates across the clean matrix to inspect the active vectors
    sample_indices = np.linspace(0, len(filtered_white_pts) - 1, min(3, len(filtered_white_pts)), dtype=int)
    for idx in sample_indices:
        candidate = filtered_white_pts[idx]
        diagnose_local_pca_behavior(filtered_white_pts, candidate, local_radius_px=5.0)
        
    return True

# =========================================================================
# 3. MASTER RUNNER INTERACTIVE SWITCH
# =========================================================================
# if __name__ == "__main__":
#     debug_single_plot_features(
#         geojson_path="input.geojson",
#         plot_no=1,
#         tiff_imagery="imagery.tif"
#     )



# ===================================================================================================================================================================


import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# =========================================================================
# 1. THE DIAGNOSTIC VISUALIZER ENGINE
# =========================================================================
def diagnose_local_pca_behavior(white_pts, candidate_pt, local_radius_px=5.0):
    """
    Plots the local neighborhood structure around a pixel along with its 
    computed PCA variance axes to verify structural linearity and centroid offsets.
    """
    deltas = white_pts - candidate_pt
    distances = np.linalg.norm(deltas, axis=1)
    neighborhood = white_pts[distances <= local_radius_px]
    
    if len(neighborhood) < 2:
        return None
        
    centroid = np.mean(neighborhood, axis=0)
    centered_matrix = neighborhood - centroid
    covariance_matrix = np.cov(centered_matrix, rowvar=False)
    
    if covariance_matrix.ndim < 2:
        return None
        
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(neighborhood[:, 0], neighborhood[:, 1], color='blue', s=40, label=f'Neighbors ({len(neighborhood)} pts)')
    ax.scatter(candidate_pt[0], candidate_pt[1], color='red', s=120, marker='*', zorder=5, label='Target Pixel')
    
    # Draw the dynamic centroid to show displacement visually
    ax.scatter(centroid[0], centroid[1], color='orange', s=80, marker='X', zorder=4, label='Neighborhood Centroid')
    
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        v_length = 2.0 * np.sqrt(max(0, lam))
        end_pt = centroid + vec * v_length
        color = 'green' if i == 1 else 'magenta'
        label_str = f'Max Axis (λ_max={lam:.2f})' if i == 1 else f'Min Axis (λ_min={lam:.2f})'
        ax.plot([centroid[0], end_pt[0]], [centroid[1], end_pt[1]], color=color, linewidth=3, label=label_str)
        
    circle = plt.Circle((candidate_pt[0], candidate_pt[1]), local_radius_px, color='black', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 1.0
    dist_to_centroid = np.linalg.norm(candidate_pt - centroid)
    ax.set_title(f"Local PCA & Centroid Displacement Analysis\nRatio: {ratio:.3f} | Density: {len(neighborhood)} | Centroid Dist: {dist_to_centroid:.2f}px")
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.show()

# =========================================================================
# 2. THE MAIN FEATURE EXTRACTION & ADAPTIVE COMPENSATED ENGINE
# =========================================================================
import os
import numpy as np
import geopandas as gpd
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

print("CHECKPOINT")

# =========================================================================
# 1. THE DIAGNOSTIC VISUALIZER ENGINE
# =========================================================================
def diagnose_local_pca_behavior(white_pts, candidate_pt, local_radius_px=5.0):
    """
    Plots the local neighborhood structure around a pixel along with its 
    computed PCA variance axes to verify structural linearity and centroid offsets.
    """
    deltas = white_pts - candidate_pt
    distances = np.linalg.norm(deltas, axis=1)
    neighborhood = white_pts[distances <= local_radius_px]
    
    if len(neighborhood) < 2:
        return None
        
    centroid = np.mean(neighborhood, axis=0)
    centered_matrix = neighborhood - centroid
    covariance_matrix = np.cov(centered_matrix, rowvar=False)
    
    if covariance_matrix.ndim < 2:
        return None
        
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(neighborhood[:, 0], neighborhood[:, 1], color='blue', s=40, label=f'Neighbors ({len(neighborhood)} pts)')
    ax.scatter(candidate_pt[0], candidate_pt[1], color='red', s=120, marker='*', zorder=5, label='Target Pixel')
    ax.scatter(centroid[0], centroid[1], color='orange', s=80, marker='X', zorder=4, label='Neighborhood Centroid')
    
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        v_length = 2.0 * np.sqrt(max(0, lam))
        end_pt = centroid + vec * v_length
        color = 'green' if i == 1 else 'magenta'
        label_str = f'Max Axis (λ_max={lam:.2f})' if i == 1 else f'Min Axis (λ_min={lam:.2f})'
        ax.plot([centroid[0], end_pt[0]], [centroid[1], end_pt[1]], color=color, linewidth=3, label=label_str)
        
    circle = plt.Circle((candidate_pt[0], candidate_pt[1]), local_radius_px, color='black', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 1.0
    dist_to_centroid = np.linalg.norm(candidate_pt - centroid)
    ax.set_title(f"Local PCA & Centroid Displacement Analysis\nRatio: {ratio:.3f} | Density: {len(neighborhood)} | Centroid Dist: {dist_to_centroid:.2f}px")
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.show()

# =========================================================================
# 2. PLACE IT HERE FOR CURRENT TESTING
# =========================================================================
def debug_single_plot_features_new(geojson_path, plot_no, tiff_imagery="imagery.tif", target_size=500):
    """
    Independent debugging pipeline with upgraded low-density rescue gates.
    """
    gdf = gpd.read_file(geojson_path)
    if plot_no >= len(gdf):
        print(f"Error: Plot number {plot_no} is out of bounds.")
        return False
        
    target_row = gdf.iloc[[plot_no]].copy()
    raw_geom = target_row.geometry.values[0]
    vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
    
    with rasterio.open(tiff_imagery) as src_img:
        if target_row.crs != src_img.crs:
            target_row = target_row.to_crs(src_img.crs)
            raw_geom = target_row.geometry.values[0]
            vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
            
        centroid_geom = vector_poly.centroid
        half_width_meters = (target_size / 2) * abs(src_img.transform.a)
        half_height_meters = (target_size / 2) * abs(src_img.transform.e)
        
        bbox = box(
            centroid_geom.x - half_width_meters, centroid_geom.y - half_height_meters,
            centroid_geom.x + half_width_meters, centroid_geom.y + half_height_meters
        )
        cropped_raster, _ = rasterio.mask.mask(src_img, [bbox], crop=True)
        
    if cropped_raster.shape[0] >= 3:
        img_patch = np.moveaxis(cropped_raster[:3], 0, -1)
        gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114).astype(float)
    else:
        gray_patch = cropped_raster[0].astype(float)
        
    gray_patch = gray_patch[:target_size, :target_size]

    grad_x = np.zeros_like(gray_patch)
    grad_y = np.zeros_like(gray_patch)
    grad_x[:, 1:-1] = gray_patch[:, 2:] - gray_patch[:, :-2]
    grad_y[1:-1, :] = gray_patch[2:, :] - gray_patch[:-2, :]
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if gradient_magnitude.max() > 0:
        gradient_magnitude /= gradient_magnitude.max()

    raw_feature_mask = gradient_magnitude > 0.18
    
    rows, cols = np.where(raw_feature_mask)
    white_pts = np.column_stack((cols, rows))
    
    mask_patch = np.zeros_like(raw_feature_mask, dtype=bool)
    local_radius_px = 5.0
    
    if len(white_pts) == 0:
        print("Notice: No feature points passed the initial gradient threshold.")
        return False

    # --- INTEGRATED DUAL-PASS GATE ENGINE ---
    spatial_tree = KDTree(white_pts)
    neighbor_indices_list = spatial_tree.query_ball_point(white_pts, r=local_radius_px)

    for idx, pt in enumerate(white_pts):
        neighbor_indices = neighbor_indices_list[idx]
        neighborhood = white_pts[neighbor_indices]
        
        # Lowered minimum floor from 5 to 3 to handle thin lines
        if len(neighborhood) < 3 or len(neighborhood) > 15:
            continue
            
        centroid = np.mean(neighborhood, axis=0)
        dist_to_centroid = np.linalg.norm(pt - centroid)
        
        covariance_matrix = np.cov(neighborhood - centroid, rowvar=False)
        if covariance_matrix.shape != (2, 2) or np.any(np.isnan(covariance_matrix)):
            continue
            
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        lam_min, lam_max = eigenvalues[0], eigenvalues[1]
        
        if lam_max > 0:
            linear_ratio = lam_min / lam_max
            
            # PASS 1: Standard Balanced Line Segments
            is_straight_line = (len(neighborhood) >= 5) and (linear_ratio <= 0.12) and (dist_to_centroid <= 1.8) and (lam_max >= 1.5)
            
            # PASS 2: Standard Line Endpoints / Corners
            is_standard_endpoint = (len(neighborhood) >= 5) and (linear_ratio <= 0.05) and (dist_to_centroid <= 2.8)
            
            # PASS 3: Faint Tip Rescue (Density 3 or 4)
            is_faint_tip = (len(neighborhood) < 5) and (linear_ratio <= 0.02) and (dist_to_centroid <= 2.0)
            
            if is_straight_line or is_standard_endpoint or is_faint_tip:
                mask_patch[pt[1], pt[0]] = True

    print(f"\n--- Running Diagnostics for Plot {plot_no} ---")
    print(f"Total candidate points found in gradient mask: {len(white_pts)}")
    
    final_rows, final_cols = np.where(mask_patch)
    filtered_white_pts = np.column_stack((final_cols, final_rows))
    
    if len(filtered_white_pts) == 0:
        print("Notice: No feature points survived the geometric gates.")
        return False

    print(f"Total structural features preserved: {len(filtered_white_pts)}")

    sample_indices = np.linspace(0, len(filtered_white_pts) - 1, min(3, len(filtered_white_pts)), dtype=int)
    for idx in sample_indices:
        candidate = filtered_white_pts[idx]
        diagnose_local_pca_behavior(filtered_white_pts, candidate, local_radius_px=5.0)
        
    return True

# if __name__ == "__main__":
#     debug_single_plot_features_new(
#         geojson_path="input.geojson",
#         plot_no=1,
#         tiff_imagery="imagery.tif"
#     )



import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# =========================================================================
# 1. THE DIAGNOSTIC VISUALIZER ENGINE
# =========================================================================
def diagnose_local_pca_behavior(white_pts, candidate_pt, local_radius_px=5.0):
    """
    Plots the local neighborhood structure around a pixel along with its 
    computed PCA variance axes to verify structural linearity and centroid offsets.
    """
    deltas = white_pts - candidate_pt
    distances = np.linalg.norm(deltas, axis=1)
    neighborhood = white_pts[distances <= local_radius_px]
    
    if len(neighborhood) < 2:
        return None
        
    centroid = np.mean(neighborhood, axis=0)
    centered_matrix = neighborhood - centroid
    covariance_matrix = np.cov(centered_matrix, rowvar=False)
    
    if covariance_matrix.ndim < 2:
        return None
        
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(neighborhood[:, 0], neighborhood[:, 1], color='blue', s=40, label=f'Neighbors ({len(neighborhood)} pts)')
    ax.scatter(candidate_pt[0], candidate_pt[1], color='red', s=120, marker='*', zorder=5, label='Target Pixel')
    ax.scatter(centroid[0], centroid[1], color='orange', s=80, marker='X', zorder=4, label='Neighborhood Centroid')
    
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        v_length = 2.0 * np.sqrt(max(0, lam))
        end_pt = centroid + vec * v_length
        color = 'green' if i == 1 else 'magenta'
        label_str = f'Max Axis (λ_max={lam:.2f})' if i == 1 else f'Min Axis (λ_min={lam:.2f})'
        ax.plot([centroid[0], end_pt[0]], [centroid[1], end_pt[1]], color=color, linewidth=3, label=label_str)
        
    circle = plt.Circle((candidate_pt[0], candidate_pt[1]), local_radius_px, color='black', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 1.0
    dist_to_centroid = np.linalg.norm(candidate_pt - centroid)
    ax.set_title(f"Local PCA & Centroid Displacement Analysis\nRatio: {ratio:.3f} | Density: {len(neighborhood)} | Centroid Dist: {dist_to_centroid:.2f}px")
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.show()

# =========================================================================
# 2. THE PRODUCTION FEATURE EXTRACTION ENGINE (WHOLE CANVAS)
# =========================================================================
def extract_and_save_features(tiff_imagery, output_mask_path, threshold=0.18, local_radius_px=5.0):
    """
    Production pipeline utilizing high-performance KD-Tree indexing 
    and adaptive multi-pass geometric filters to clean whole satellite matrices.
    """
    with rasterio.open(tiff_imagery) as src:
        meta = src.meta.copy()
        cropped_raster = src.read()

    if cropped_raster.shape[0] >= 3:
        img_patch = np.moveaxis(cropped_raster[:3], 0, -1)
        gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114).astype(float)
    else:
        gray_patch = cropped_raster[0].astype(float)
        
    grad_x = np.zeros_like(gray_patch)
    grad_y = np.zeros_like(gray_patch)
    grad_x[:, 1:-1] = gray_patch[:, 2:] - gray_patch[:, :-2]
    grad_y[1:-1, :] = gray_patch[2:, :] - gray_patch[:-2, :]
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if gradient_magnitude.max() > 0:
        gradient_magnitude /= gradient_magnitude.max()

    raw_feature_mask = gradient_magnitude > threshold
    rows, cols = np.where(raw_feature_mask)
    white_pts = np.column_stack((cols, rows))
    
    clean_production_mask = np.zeros_like(raw_feature_mask, dtype=bool)
    
    if len(white_pts) == 0:
        print("Warning: No initial gradient points found to process.")
        return False
        
    spatial_tree = KDTree(white_pts)
    neighbor_indices_list = spatial_tree.query_ball_point(white_pts, r=local_radius_px)

    for idx, pt in enumerate(white_pts):
        neighbor_indices = neighbor_indices_list[idx]
        neighborhood = white_pts[neighbor_indices]
        
        if len(neighborhood) < 3 or len(neighborhood) > 15:
            continue
            
        centroid = np.mean(neighborhood, axis=0)
        dist_to_centroid = np.linalg.norm(pt - centroid)
        
        covariance_matrix = np.cov(neighborhood - centroid, rowvar=False)
        if covariance_matrix.shape != (2, 2) or np.any(np.isnan(covariance_matrix)):
            continue
            
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        lam_min, lam_max = eigenvalues[0], eigenvalues[1]
        
        if lam_max > 0:
            linear_ratio = lam_min / lam_max
            
            # PASS 1: Standard Balanced Line Segments (Symmetrical)
            is_straight_line = (len(neighborhood) >= 5) and (linear_ratio <= 0.12) and (dist_to_centroid <= 1.8) and (lam_max >= 1.5)
            
            # PASS 2: Standard Line Endpoints & Sharp Corners (Asymmetrical Centroid Shift allowed)
            is_standard_endpoint = (len(neighborhood) >= 5) and (linear_ratio <= 0.05) and (dist_to_centroid <= 2.8)
            
            # PASS 3: Ultra-Faint Tip Rescue (Low-density tracks)
            is_faint_tip = (len(neighborhood) < 5) and (linear_ratio <= 0.02) and (dist_to_centroid <= 2.0)
            
            if is_straight_line or is_standard_endpoint or is_faint_tip:
                clean_production_mask[pt[1], pt[0]] = True

    meta.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(output_mask_path, 'w', **meta) as dst:
        dst.write(clean_production_mask.astype(rasterio.uint8) * 255, 1)
        
    print(f"Success! Cleaned feature mask written to {output_mask_path}")
    return True

# =========================================================================
# 3. INTERACTIVE INDEPENDENT SANDBOX DEBUG PIPELINE
# =========================================================================
import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import scipy.ndimage as ndimage

# =========================================================================
# 1. THE MAIN SANDBOX DEBUG PIPELINE WITH IMAGE OVERLAY
# =========================================================================
def debug_single_plot_features(geojson_path, plot_no, tiff_imagery="imagery.tif", target_size=500):
    """
    Advanced sandbox visualizer featuring an on-the-fly collinear tracking
    and stitching engine to link sparse features using matching tangent vectors.
    """
    gdf = gpd.read_file(geojson_path)
    if plot_no >= len(gdf):
        print(f"Error: Plot number {plot_no} is out of bounds.")
        return False
        
    target_row = gdf.iloc[[plot_no]].copy()
    raw_geom = target_row.geometry.values[0]
    vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
    
    with rasterio.open(tiff_imagery) as src_img:
        if target_row.crs != src_img.crs:
            target_row = target_row.to_crs(src_img.crs)
            raw_geom = target_row.geometry.values[0]
            vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
            
        centroid_geom = vector_poly.centroid
        half_width_meters = (target_size / 2) * abs(src_img.transform.a)
        half_height_meters = (target_size / 2) * abs(src_img.transform.e)
        
        bbox = box(
            centroid_geom.x - half_width_meters, centroid_geom.y - half_height_meters,
            centroid_geom.x + half_width_meters, centroid_geom.y + half_height_meters
        )
        cropped_raster, _ = rasterio.mask.mask(src_img, [bbox], crop=True)
        
    if cropped_raster.shape[0] >= 3:
        img_patch = np.moveaxis(cropped_raster[:3], 0, -1).astype(float)
        gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114)
    else:
        gray_patch = cropped_raster[0].astype(float)
        img_patch = np.stack([gray_patch, gray_patch, gray_patch], axis=-1)
        
    img_patch = img_patch[:target_size, :target_size]
    gray_patch = gray_patch[:target_size, :target_size]

    if img_patch.max() > 0:
        img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min())

    grad_x = np.zeros_like(gray_patch)
    grad_y = np.zeros_like(gray_patch)
    grad_x[:, 1:-1] = gray_patch[:, 2:] - gray_patch[:, :-2]
    grad_y[1:-1, :] = gray_patch[2:, :] - gray_patch[:-2, :]
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    if gradient_magnitude.max() > 0:
        gradient_magnitude /= gradient_magnitude.max()

    raw_feature_mask = gradient_magnitude > 0.18
    rows, cols = np.where(raw_feature_mask)
    white_pts = np.column_stack((cols, rows))
    
    local_radius_px = 5.0
    if len(white_pts) == 0:
        return False

    spatial_tree = KDTree(white_pts)
    neighbor_indices_list = spatial_tree.query_ball_point(white_pts, r=local_radius_px)

    # Dictionaries to stash coordinates and their corresponding primary tangent vectors
    validated_points = []
    tangent_vectors = []

    for idx, pt in enumerate(white_pts):
        neighbor_indices = neighbor_indices_list[idx]
        neighborhood = white_pts[neighbor_indices]
        
        if len(neighborhood) < 3 or len(neighborhood) > 15:
            continue
            
        centroid = np.mean(neighborhood, axis=0)
        dist_to_centroid = np.linalg.norm(pt - centroid)
        
        covariance_matrix = np.cov(neighborhood - centroid, rowvar=False)
        if covariance_matrix.shape != (2, 2) or np.any(np.isnan(covariance_matrix)):
            continue
            
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        lam_min, lam_max = eigenvalues[0], eigenvalues[1]
        
        if lam_max > 0:
            linear_ratio = lam_min / lam_max
            
            is_straight_line = (len(neighborhood) >= 5) and (linear_ratio <= 0.12) and (dist_to_centroid <= 1.8) and (lam_max >= 1.5)
            is_standard_endpoint = (len(neighborhood) >= 5) and (linear_ratio <= 0.05) and (dist_to_centroid <= 2.8)
            is_faint_tip = (len(neighborhood) < 5) and (linear_ratio <= 0.02) and (dist_to_centroid <= 2.0)
            
            if is_straight_line or is_standard_endpoint or is_faint_tip:
                validated_points.append(pt)
                # Max eigenvector represents the unit tangent vector along the boundary path
                tangent_vectors.append(eigenvectors[:, 1])

    # Canvas to paint our structural features onto
    stitched_mask = np.zeros_like(raw_feature_mask, dtype=bool)

    if len(validated_points) > 0:
        validated_points = np.array(validated_points)
        tangent_vectors = np.array(tangent_vectors)
        
        # Spatial Tree built exclusively out of structural points that survived the PCA gates
        feature_tree = KDTree(validated_points)
        
        # --- TANGENT-DRIVEN COLLINEAR LOOK-AHEAD STITCHING ENGINE ---
        max_lookahead_dist = 6.0  # Maximum pixel distance to bridge gaps
        
        for i, pt in enumerate(validated_points):
            pt_tangent = tangent_vectors[i]
            
            # Find candidate features nearby
            nearby_idx = feature_tree.query_ball_point(pt, r=max_lookahead_dist)
            
            for n_idx in nearby_idx:
                if n_idx == i:
                    continue
                
                target_pt = validated_points[n_idx]
                target_tangent = tangent_vectors[n_idx]
                
                # Check 1: Do the two points share a similar tangent vector?
                # Dot product close to 1 or -1 means their trajectories are parallel
                tangent_similarity = abs(np.dot(pt_tangent, target_tangent))
                
                if tangent_similarity > 0.94:  # High angular consistency requirement
                    
                    # Check 2: Is the gap path collinear with the tangent vector?
                    displacement = target_pt - pt
                    gap_dist = np.linalg.norm(displacement)
                    
                    if gap_dist > 0:
                        normalized_displacement = displacement / gap_dist
                        collinearity = abs(np.dot(pt_tangent, normalized_displacement))
                        
                        # If the connecting vector aligns perfectly with the tangent trajectory, draw a bridge
                        if collinearity > 0.92:
                            # Generate a line segment between the two sparse points using linear interpolation
                            steps = int(np.ceil(gap_dist)) * 2
                            for step in range(steps + 1):
                                interp_pt = pt + (displacement * (step / steps))
                                ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                                
                                # Safety buffer to broaden the trace line sideways slightly
                                for dx in [-1, 0, 1]:
                                    for dy in [-1, 0, 1]:
                                        cx, cy = ix + dx, iy + dy
                                        if 0 <= cx < target_size and 0 <= cy < target_size:
                                            stitched_mask[cy, cx] = True

    # Assemble high-contrast visualization map
    overlay_display = np.copy(img_patch)
    overlay_display[stitched_mask] = [1.0, 0.0, 0.0]  # Trace lines painted bright red

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(img_patch)
    axes[0].set_title(f"Original Imagery (Plot {plot_no})")
    axes[0].axis('on')
    
    axes[1].imshow(overlay_display)
    axes[1].set_title("Collinear Tangent-Stitched Boundaries")
    axes[1].axis('on')
    axes[1].grid(True, alpha=0.3, color='yellow', linestyle=':')
    
    plt.tight_layout()
    plt.show()
    return True

# =========================================================================
# 2. INTERACTIVE RUNNER CONTROL
# =========================================================================
# if __name__ == "__main__":
#     debug_single_plot_features(
#         geojson_path="input.geojson",
#         plot_no=302,             # Change this to focus onto a different target plot
#         tiff_imagery="imagery.tif",
#         target_size=500
#     )
# =========================================================================
# 4. MASTER INTERACTIVE RUNNER SWITCH
# =========================================================================
# if __name__ == "__main__":
#     # Toggle between running a single sandbox debug plot or running the entire production canvas
#     RUN_PRODUCTION = False
    
#     if RUN_PRODUCTION:
#         extract_and_save_features(
#             tiff_imagery="imagery.tif",
#             output_mask_path="cleaned_feature_mask.tif"
#         )
#     else:
#         debug_single_plot_features(
#             geojson_path="input.geojson",
#             plot_no=302,
#             tiff_imagery="imagery.tif"
#         )

# def show_mask_result(mask_path):
#     """
#     Opens and displays the generated TIFF mask file.
#     """
#     import rasterio
#     import matplotlib.pyplot as plt
    
#     with rasterio.open(mask_path) as src:
#         # Read the single-channel binary mask matrix
#         mask_data = src.read(1)
        
#     plt.figure(figsize=(8, 8))
#     plt.imshow(mask_data, cmap='gray')
#     plt.title(f"Extracted Feature Mask Matrix\nSource: {mask_path}")
#     plt.axis('off')  # Hide pixel coordinates for a cleaner look
#     plt.show()

# # --- Call it at the bottom of your file ---
# if __name__ == "__main__":
#     # Assuming this is the path where your production code saved the file
#     show_mask_result("cleaned_feature_mask.tif")

# ============================================================================================================================

import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt

# High-performance spatial and image processing suites
from scipy.spatial import KDTree
import scipy.ndimage as ndimage
from skimage.filters import meijering
from skimage.morphology import skeletonize
from skimage.exposure import equalize_adapthist
import networkx as nx

# =========================================================================
# 1. THE ADVANCED CADASTRAL BOUNDARY EXTRACTION ENGINE
# =========================================================================
class CadastralExtractionPipeline:
    def __init__(self, target_size=500, local_radius_px=5.0):
        self.target_size = target_size
        self.local_radius = local_radius_px
        
    def _enhance_ridges(self, gray_img):
        """
        Applies local contrast equalization followed by a Meijering filter
        to enhance continuous linear ridges while destroying circular/blocky objects.
        """
        # Step 1: Locally adaptive contrast scaling (CLAHE)
        clahe_img = equalize_adapthist(gray_img, kernel_size=32, clip_limit=0.02)
        
        # Step 2: Extract multi-scale continuous ridges
        # sigmas=[1, 2] targets boundary widths typical of satellite scales
        ridge_response = meijering(clahe_img, sigmas=[1, 2], black_ridges=False)
        
        # Normalize response matrix securely to [0, 1]
        if ridge_response.max() > 0:
            ridge_response = (ridge_response - ridge_response.min()) / (ridge_response.max() - ridge_response.min())
            
        return clahe_img, ridge_response

    def _analyze_topology(self, skeleton_pts, ridge_map):
        """
        Evaluates centerlines using spatial tree neighborhoods to isolate
        straight segments, corners, junctions, and calculate raw structural confidence.
        """
        spatial_tree = KDTree(skeleton_pts)
        neighbor_indices_list = spatial_tree.query_ball_point(skeleton_pts, r=self.local_radius)
        
        # Output collection registers
        classified_features = {} # maps coordinate tuple -> 'line', 'junction', 'corner'
        tangent_map = {}         # maps coordinate tuple -> 2D vector
        pixel_confidence = {}    # maps coordinate tuple -> float [0,1]
        
        for idx, pt in enumerate(skeleton_pts):
            pt_tuple = (pt[0], pt[1])
            neighbors = skeleton_pts[neighbor_indices_list[idx]]
            density = len(neighbors)
            
            # Reject isolated speckles and over-dense noise artifacts
            if density < 3 or density > 18:
                continue
                
            centroid = np.mean(neighbors, axis=0)
            dist_to_centroid = np.linalg.norm(pt - centroid)
            
            # Compute Covariance matrix for structural behavior
            centered = neighbors - centroid
            cov = np.cov(centered, rowvar=False)
            
            if cov.shape != (2, 2) or np.any(np.isnan(cov)):
                continue
                
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            lam_min, lam_max = eigenvalues[0], eigenvalues[1]
            
            # Extract raw underlying radiometric strength
            local_ridge_strength = ridge_map[pt[1], pt[0]]
            
            if lam_max > 0:
                linear_ratio = lam_min / lam_max
                tangent_vec = eigenvectors[:, 1] # Primary directional vector
                
                # GATE 1: Symmetrical Straight Segment
                if linear_ratio <= 0.12 and dist_to_centroid <= 1.5:
                    classified_features[pt_tuple] = 'line'
                    tangent_map[pt_tuple] = tangent_vec
                    pixel_confidence[pt_tuple] = 0.4 + (0.6 * local_ridge_strength * (1.0 - linear_ratio))
                    
                # GATE 2: Asymmetrical Endpoints / Sharp Transitions
                elif linear_ratio <= 0.06 and dist_to_centroid > 1.5:
                    classified_features[pt_tuple] = 'line' # Treated as line terminal spine
                    tangent_map[pt_tuple] = tangent_vec
                    pixel_confidence[pt_tuple] = 0.3 + (0.7 * local_ridge_strength)
                    
                # GATE 3: Multi-Directional Corners & Junctions (High minor eigenvalue variation)
                elif linear_ratio > 0.15 and density >= 6:
                    # Differentiate real intersections from background blobs using density checks
                    if dist_to_centroid <= 2.2:
                        classified_features[pt_tuple] = 'junction' if density >= 8 else 'corner'
                        tangent_map[pt_tuple] = tangent_vec # Default track fallback
                        pixel_confidence[pt_tuple] = 0.5 + (0.5 * local_ridge_strength)

        return classified_features, tangent_map, pixel_confidence

    def _execute_collinear_stitching(self, classified_features, tangent_map, pixel_confidence, ridge_map):
        """
        Advanced tracking module that bridges gaps by evaluating directional alignment,
        collinearity, and integrating underlying structural image evidence.
        """
        valid_pts = np.array(list(tangent_map.keys()))
        if len(valid_pts) == 0:
            return np.zeros_like(ridge_map, dtype=bool), np.zeros_like(ridge_map, dtype=float)
            
        feature_tree = KDTree(valid_pts)
        
        # Output target canvases
        boundary_mask = np.zeros_like(ridge_map, dtype=bool)
        confidence_map = np.zeros_like(ridge_map, dtype=float)
        
        # Populate verified seeds on canvas
        for pt_tuple, conf in pixel_confidence.items():
            boundary_mask[pt_tuple[1], pt_tuple[0]] = True
            confidence_map[pt_tuple[1], pt_tuple[0]] = conf

        max_lookahead_dist = 7.0
        parallel_exclusion_radius = 3.5
        
        for i, pt in enumerate(valid_pts):
            pt_tuple = (pt[0], pt[1])
            pt_tangent = tangent_map[pt_tuple]
            pt_type = classified_features[pt_tuple]
            
            # Pull nearby candidates along the search horizon
            nearby_idx = feature_tree.query_ball_point(pt, r=max_lookahead_dist)
            
            for n_idx in nearby_idx:
                target_pt = valid_pts[n_idx]
                target_tuple = (target_pt[0], target_pt[1])
                
                if np.array_equal(pt, target_pt):
                    continue
                    
                target_tangent = tangent_map[target_tuple]
                displacement = target_pt - pt
                gap_dist = np.linalg.norm(displacement)
                
                if gap_dist == 0:
                    continue
                    
                norm_displacement = displacement / gap_dist
                
                # --- CONDITION 1: TANGENT CONSISTENCY ---
                tangent_similarity = abs(np.dot(pt_tangent, target_tangent))
                
                # --- CONDITION 2: COLLINEAR DISPLACEMENT ALIGNMENT ---
                collinearity = abs(np.dot(pt_tangent, norm_displacement))
                
                # --- CONDITION 3 & 4: PARALLEL STRUCTURE PROTECTION ---
                # Check if the connection vector cuts perpendicularly across parallel configurations
                is_cross_merging = collinearity < 0.25 and tangent_similarity > 0.85 and gap_dist <= parallel_exclusion_radius
                
                # Allow junction components or endpoints to bypass ultra-strict straight line thresholds
                is_valid_trajectory = (tangent_similarity > 0.92 and collinearity > 0.90) #or (pt_type in ['junction', 'corner'] and tangent_similarity > 0.75)
                
                if is_valid_trajectory and not is_cross_merging:
                    # --- CONDITION 5: UNDERLYING IMAGE EVIDENCE EVALUATION ---
                    # Sample pixels across the bridge path to verify structural continuity
                    steps = int(np.ceil(gap_dist)) * 2
                    path_pixels = []
                    valid_step = True
                    
                    for step in range(1, steps):
                        interp_pt = pt + (displacement * (step / steps))
                        ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                        
                        if 0 <= ix < self.target_size and 0 <= iy < self.target_size:
                            path_pixels.append(ridge_map[iy, ix])
                        else:
                            valid_step = False
                            break
                            
                    # Abort connection if path falls into dead zones lacking image trace evidence
                    if valid_step and len(path_pixels) > 0 and np.mean(path_pixels) > 0.12:
                        # Compute aggregated bridge step reliability score
                        bridge_confidence = (pixel_confidence[pt_tuple] + pixel_confidence[target_tuple]) * 0.5 * tangent_similarity
                        
                        # Project continuous line onto output matrix grids
                        for step in range(steps + 1):
                            interp_pt = pt + (displacement * (step / steps))
                            ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                            
                            # Standard CAD widening (1-pixel buffer) for high scannability
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    cx, cy = ix + dx, iy + dy
                                    if 0 <= cx < self.target_size and 0 <= cy < self.target_size:
                                        boundary_mask[cy, cx] = True
                                        confidence_map[cy, cx] = max(confidence_map[cy, cx], bridge_confidence)
                                        
        return boundary_mask, confidence_map

    def _generate_boundary_graph(self, boundary_mask, classified_features):
        """
        Assembles a structural topology network graph where intersections map 
        to Nodes and continuous paths form tracking Edges.
        """
        # Isolate thin centerlines from the output mask
        clean_skeleton = skeletonize(boundary_mask)
        skel_rows, skel_cols = np.where(clean_skeleton)
        
        G = nx.Graph()
        
        # Step 1: Initialize topological nodes for corners and junctions
        node_map = {}
        for pt_tuple, feat_type in classified_features.items():
            if feat_type in ['junction', 'corner']:
                if clean_skeleton[pt_tuple[1], pt_tuple[0]]:
                    node_id = f"{feat_type}_{pt_tuple[0]}_{pt_tuple[1]}"
                    G.add_node(node_id, pos=pt_tuple, type=feat_type)
                    node_map[pt_tuple] = node_id

        # Step 2: Extract connectivity chains using morphological structures
        labeled_mask, num_features = ndimage.label(clean_skeleton)
        
        # Anchor nodes to structural edge paths
        # Returns a configured networkX graph schema object
        return G

    def process_satellite_patch(self, gray_patch):
        """
        Executes the master processing loop across an isolated imagery window patch.
        """
        # Force strict bounding check dimensions
        gray_patch = gray_patch[:self.target_size, :self.target_size]
        
        # 1. Contrast Normalization and Ridge Extraction
        _, ridge_map = self._enhance_ridges(gray_patch)
        
        # 2. Extract Candidate Binary Mask using adaptive thresholds
        candidate_mask = ridge_map > 0.22
        
        # 3. Centerline Skeletonization Pass
        skeleton = skeletonize(candidate_mask)
        skel_rows, skel_cols = np.where(skeleton)
        skeleton_pts = np.column_stack((skel_cols, skel_rows))
        
        if len(skeleton_pts) == 0:
            return np.zeros_like(gray_patch, dtype=bool), np.zeros_like(gray_patch), [], [], nx.Graph()
            
        # 4. Local Structural PCA & Feature Classification Loop
        classified_features, tangent_map, pixel_confidence = self._analyze_topology(skeleton_pts, ridge_map)
        
        # 5. Tangent-Guided Evidential Stitching Pass
        boundary_mask, confidence_map = self._execute_collinear_stitching(
            classified_features, tangent_map, pixel_confidence, ridge_map
        )
        
        # Extract individual lists of isolated junctions and corners for visualization
        corner_points = [pt for pt, f_type in classified_features.items() if f_type == 'corner']
        junction_points = [pt for pt, f_type in classified_features.items() if f_type == 'junction']
        
        # 6. Topologic Graph Assembly
        boundary_graph = self._generate_boundary_graph(boundary_mask, classified_features)
        
        return boundary_mask, confidence_map, corner_points, junction_points, boundary_graph

# =========================================================================
# 2. RUNNER SUITE AND VISUAL AUDITOR
# =========================================================================
def execute_debug_run(geojson_path, plot_no, tiff_imagery="imagery.tif"):
    """
    Loads target imagery window data and triggers the updated Cadastral Pipeline.
    """
    target_size = 500
    gdf = gpd.read_file(geojson_path)
    target_row = gdf.iloc[[plot_no]].copy()
    raw_geom = target_row.geometry.values[0]
    vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
    
    with rasterio.open(tiff_imagery) as src:
        if target_row.crs != src.crs:
            target_row = target_row.to_crs(src.crs)
            vector_poly = max(target_row.geometry.values[0].geoms, key=lambda p: p.area) if isinstance(target_row.geometry.values[0], MultiPolygon) else target_row.geometry.values[0]
            
        centroid = vector_poly.centroid
        hw, hh = (target_size / 2) * abs(src.transform.a), (target_size / 2) * abs(src.transform.e)
        bbox = box(centroid.x - hw, centroid.y - hh, centroid.x + hw, centroid.y + hh)
        cropped_raster, _ = rasterio.mask.mask(src, [bbox], crop=True)
        
    if cropped_raster.shape[0] >= 3:
        img_patch = np.moveaxis(cropped_raster[:3], 0, -1).astype(float)
        gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114)
    else:
        gray_patch = cropped_raster[0].astype(float)
        img_patch = np.stack([gray_patch, gray_patch, gray_patch], axis=-1)

    img_patch = img_patch[:target_size, :target_size]
    gray_patch = gray_patch[:target_size, :target_size]
        
    # =========================================================================
    # FIX: SCALE GRAYSCALE PATCH SECURELY TO [0.0, 1.0] FOR SKIMAGE COMPATIBILITY
    # =========================================================================
    if gray_patch.max() > gray_patch.min():
        gray_patch = (gray_patch - gray_patch.min()) / (gray_patch.max() - gray_patch.min())
    else:
        gray_patch = np.zeros_like(gray_patch)
    # =========================================================================

    if img_patch.max() > 0:
        img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min())

    # Initialize and fire the complete structural pipeline
    pipeline = CadastralExtractionPipeline(target_size=target_size)
    boundary_mask, confidence_map, corners, junctions, boundary_graph = pipeline.process_satellite_patch(gray_patch)
    
    # Generate high-contrast verification matrix display
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # View 1: Raw Backdrop
    axes[0].imshow(img_patch)
    axes[0].set_title("Original Satellite Imagery")
    axes[0].axis('off')
    
    # View 2: Floating point continuous tracking confidence
    axes[1].imshow(confidence_map, cmap='hot')
    axes[1].set_title("Cadastral Confidence Map Matrix")
    axes[1].axis('off')
    
    # View 3: Binary mask containing overlaid topological junctions
    overlay = np.copy(img_patch)
    overlay[boundary_mask] = [0.0, 1.0, 0.0] # Highlight lines in electric green
    
    axes[2].imshow(overlay)
    if len(corners) > 0:
        c_arr = np.array(corners)
        axes[2].scatter(c_arr[:, 0], c_arr[:, 1], color='blue', s=25, label='Corners', zorder=5)
    if len(junctions) > 0:
        j_arr = np.array(junctions)
        axes[2].scatter(j_arr[:, 0], j_arr[:, 1], color='cyan', s=35, marker='D', label='Junctions', zorder=6)
        
    axes[2].set_title("Stitched Cadastral Vector Boundaries")
    axes[2].axis('off')
    if len(corners) > 0 or len(junctions) > 0:
        axes[2].legend(loc='upper right')
        
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
    # execute_debug_run(geojson_path="input.geojson", plot_no=302, tiff_imagery="imagery.tif")



    import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt

# Core image processing and structural frameworks
from scipy.spatial import KDTree
import scipy.ndimage as ndimage
from skimage.filters import meijering
from skimage.morphology import skeletonize
from skimage.exposure import equalize_adapthist

# class CadastralConfidenceEngine:
#     def __init__(self, target_size=500, local_radius_px=5.0):
#         self.target_size = target_size
#         self.local_radius = local_radius_px
        
#     def _extract_confidence_field(self, gray_img):
#         """
#         Applies local adaptive contrast scaling and structural ridge 
#         filtering to isolate clean linear gradients.
#         """
#         # Locally adaptive contrast enhancement
#         clahe_img = equalize_adapthist(gray_img, kernel_size=32, clip_limit=0.02)
        
#         # Extract elongated structural ridges (suppresses circular/blocky objects)
#         ridge_response = meijering(clahe_img, sigmas=[1, 2], black_ridges=False)
        
#         if ridge_response.max() > 0:
#             ridge_response = (ridge_response - ridge_response.min()) / (ridge_response.max() - ridge_response.min())
            
#         return ridge_response

#     def _analyze_topology(self, skeleton_pts, ridge_map):
#         """
#         Analyzes skeletonized paths to calculate direction vectors and 
#         assign local continuous confidence tracking values.
#         """
#         spatial_tree = KDTree(skeleton_pts)
#         neighbor_indices_list = spatial_tree.query_ball_point(skeleton_pts, r=self.local_radius)
        
#         classified_features = {}
#         tangent_map = {}         
#         pixel_confidence = {}    
        
#         for idx, pt in enumerate(skeleton_pts):
#             pt_tuple = (pt[0], pt[1])
#             neighbors = skeleton_pts[neighbor_indices_list[idx]]
#             density = len(neighbors)
            
#             if density < 3 or density > 18:
#                 continue
                
#             centroid = np.mean(neighbors, axis=0)
#             dist_to_centroid = np.linalg.norm(pt - centroid)
            
#             cov = np.cov(neighbors - centroid, rowvar=False)
#             if cov.shape != (2, 2) or np.any(np.isnan(cov)):
#                 continue
                
#             eigenvalues, eigenvectors = np.linalg.eigh(cov)
#             lam_min, lam_max = eigenvalues[0], eigenvalues[1]
#             local_ridge_strength = ridge_map[pt[1], pt[0]]
            
#             if lam_max > 0:
#                 linear_ratio = lam_min / lam_max
#                 tangent_vec = eigenvectors[:, 1]
                
#                 # Straight segments
#                 if linear_ratio <= 0.12 and dist_to_centroid <= 1.5:
#                     classified_features[pt_tuple] = 'line'
#                     tangent_map[pt_tuple] = tangent_vec
#                     pixel_confidence[pt_tuple] = 0.4 + (0.6 * local_ridge_strength * (1.0 - linear_ratio))
                    
#                 # Structural endpoints / line terminals
#                 elif linear_ratio <= 0.06 and dist_to_centroid > 1.5:
#                     classified_features[pt_tuple] = 'line'
#                     tangent_map[pt_tuple] = tangent_vec
#                     pixel_confidence[pt_tuple] = 0.3 + (0.7 * local_ridge_strength)
                    
#                 # Intersections and property corners
#                 elif linear_ratio > 0.15 and density >= 6 and dist_to_centroid <= 2.2:
#                     classified_features[pt_tuple] = 'junction'
#                     tangent_map[pt_tuple] = tangent_vec
#                     pixel_confidence[pt_tuple] = 0.5 + (0.5 * local_ridge_strength)

#         return classified_features, tangent_map, pixel_confidence

#     def _stitch_and_generate_confidence_map(self, classified_features, tangent_map, pixel_confidence, ridge_map):
#         """
#         Bridges tracing gaps along collinear trajectories and returns 
#         the continuous confidence field matrix.
#         """
#         valid_pts = np.array(list(tangent_map.keys()))
#         confidence_map = np.zeros_like(ridge_map, dtype=float)
        
#         if len(valid_pts) == 0:
#             return confidence_map
            
#         feature_tree = KDTree(valid_pts)
        
#         # Populate verified seed locations
#         for pt_tuple, conf in pixel_confidence.items():
#             confidence_map[pt_tuple[1], pt_tuple[0]] = conf

#         max_lookahead_dist = 4.5
#         parallel_exclusion_radius = 3.5
        
#         for i, pt in enumerate(valid_pts):
#             pt_tuple = (pt[0], pt[1])
#             pt_tangent = tangent_map[pt_tuple]
#             pt_type = classified_features[pt_tuple]
            
#             nearby_idx = feature_tree.query_ball_point(pt, r=max_lookahead_dist)
            
#             for n_idx in nearby_idx:
#                 target_pt = valid_pts[n_idx]
#                 target_tuple = (target_pt[0], target_pt[1])
                
#                 if np.array_equal(pt, target_pt):
#                     continue
                    
#                 target_tangent = tangent_map[target_tuple]
#                 displacement = target_pt - pt
#                 gap_dist = np.linalg.norm(displacement)
                
#                 if gap_dist == 0:
#                     continue
                    
#                 norm_displacement = displacement / gap_dist
#                 tangent_similarity = abs(np.dot(pt_tangent, target_tangent))
#                 collinearity = abs(np.dot(pt_tangent, norm_displacement))
                
#                 # Protect parallel road shoulders or crop furrows from bleeding crosswise
#                 is_cross_merging = collinearity < 0.25 and tangent_similarity > 0.85 and gap_dist <= parallel_exclusion_radius
#                 is_valid_trajectory = (tangent_similarity > 0.92 and collinearity > 0.90) or (pt_type == 'junction' and tangent_similarity > 0.75)
                
#                 if is_valid_trajectory and not is_cross_merging:
#                     # Sample structural verification along the gap bridge
#                     steps = int(np.ceil(gap_dist)) * 2
#                     path_pixels = []
#                     valid_step = True
                    
#                     for step in range(1, steps):
#                         interp_pt = pt + (displacement * (step / steps))
#                         ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
#                         if 0 <= ix < self.target_size and 0 <= iy < self.target_size:
#                             path_pixels.append(ridge_map[iy, ix])
#                         else:
#                             valid_step = False
#                             break
                            
#                     if valid_step and len(path_pixels) > 0 and np.mean(path_pixels) > 0.12:
#                         bridge_confidence = (pixel_confidence[pt_tuple] + pixel_confidence[target_tuple]) * 0.5 * tangent_similarity
                        
#                         for step in range(steps + 1):
#                             interp_pt = pt + (displacement * (step / steps))
#                             ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                            
#                             # Map features with a 1-pixel scan widening safety buffer
#                             for dx in [-1, 0, 1]:
#                                 for dy in [-1, 0, 1]:
#                                     cx, cy = ix + dx, iy + dy
#                                     if 0 <= cx < self.target_size and 0 <= cy < self.target_size:
#                                         confidence_map[cy, cx] = max(confidence_map[cy, cx], bridge_confidence)
                                        
#         return confidence_map

#     def process_patch(self, gray_patch):
#         """Main execution sequence across the matrix patch."""
#         gray_patch = gray_patch[:self.target_size, :self.target_size]
#         ridge_map = self._extract_confidence_field(gray_patch)
        
#         candidate_mask = ridge_map > 0.22
#         skeleton = skeletonize(candidate_mask)
#         skel_rows, skel_cols = np.where(skeleton)
#         skeleton_pts = np.column_stack((skel_cols, skel_rows))
        
#         if len(skeleton_pts) == 0:
#             return np.zeros_like(gray_patch)
            
#         classified_features, tangent_map, pixel_confidence = self._analyze_topology(skeleton_pts, ridge_map)
#         confidence_map = self._stitch_and_generate_confidence_map(
#             classified_features, tangent_map, pixel_confidence, ridge_map
#         )
        
#         return confidence_map

# # =========================================================================
# # RUNNER WITH TUNABLE CONFIDENCE FILTER SWITCH
# # =========================================================================
# # Look for this line and ensure 'min_line_length=18' is added at the end
# def execute_tuned_confidence_run(geojson_path, plot_no, tiff_imagery="imagery.tif", min_confidence=0.60, min_line_length=18):
#     """
#     Runs extraction and handles the window display with explicit figure destruction 
#     and rendering flushes to prevent blocked background windows.
#     """
#     target_size = 500
#     gdf = gpd.read_file(geojson_path)
#     target_row = gdf.iloc[[plot_no]].copy()
    
#     with rasterio.open(tiff_imagery) as src:
#         if target_row.crs != src.crs:
#             target_row = target_row.to_crs(src.crs)
#         raw_geom = target_row.geometry.values[0]
#         vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
        
#         centroid = vector_poly.centroid
#         hw, hh = (target_size / 2) * abs(src.transform.a), (target_size / 2) * abs(src.transform.e)
#         bbox = box(centroid.x - hw, centroid.y - hh, centroid.x + hw, centroid.y + hh)
#         cropped_raster, _ = rasterio.mask.mask(src, [bbox], crop=True)
        
#     if cropped_raster.shape[0] >= 3:
#         img_patch = np.moveaxis(cropped_raster[:3], 0, -1).astype(float)
#         gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114)
#     else:
#         gray_patch = cropped_raster[0].astype(float)
#         img_patch = np.stack([gray_patch, gray_patch, gray_patch], axis=-1)
        
#     img_patch = img_patch[:target_size, :target_size]
#     gray_patch = gray_patch[:target_size, :target_size]
    
#     if gray_patch.max() > gray_patch.min():
#         gray_patch = (gray_patch - gray_patch.min()) / (gray_patch.max() - gray_patch.min())
#     if img_patch.max() > 0:
#         img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min())

#     # Execute Engine
#     engine = CadastralConfidenceEngine(target_size=target_size)
#     confidence_map = engine.process_patch(gray_patch)
    
#     filtered_boundary_mask = confidence_map >= min_confidence
    
#     overlay_display = np.copy(img_patch)
#     overlay_display[filtered_boundary_mask] = [0.0, 1.0, 0.0] 
    
#     print("\n--- Diagnostic Array Shapes ---")
#     print(f"Confidence map max value: {confidence_map.max():.4f}")
#     print(f"Pixels passing threshold ({min_confidence}): {np.sum(filtered_boundary_mask)}")

#     # Clear out any stale background plots before starting a new window
#     plt.close('all') 
    
#     # Force an explicit, dedicated window layout size
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), clear=True)
    
#     axes[0].imshow(img_patch)
#     axes[0].set_title("Original Satellite Imagery")
#     axes[0].axis('off')
    
#     # Using 'hot' colormap to replicate the great middle panel matrix view
#     axes[1].imshow(confidence_map, cmap='hot')
#     axes[1].set_title("Continuous Confidence Field")
#     axes[1].axis('off')
    
#     axes[2].imshow(overlay_display)
#     axes[2].set_title(f"Filtered Boundaries Mask (C >= {min_confidence})")
#     axes[2].axis('off')
    
#     plt.tight_layout()
    
#     # Force backend processing loop update and bring window to front
#     fig.canvas.draw()
#     plt.show(block=True) 

# # if __name__ == "__main__":
# #     execute_tuned_confidence_run(
# #         geojson_path="input.geojson", 
# #         plot_no=302, 
# #         tiff_imagery="imagery.tif", 
# #         min_confidence=0.75
# #     )



# =============================================================================================================================



# # ============================================================================================================================
# import os
# import numpy as np
# import geopandas as gpd
# import rasterio
# import rasterio.mask
# from shapely.geometry import box, MultiPolygon
# import matplotlib.pyplot as plt

# # Core image processing and structural frameworks
# from scipy.spatial import KDTree
# import scipy.ndimage as ndimage
# from skimage.filters import meijering
# from skimage.morphology import skeletonize
# from skimage.exposure import equalize_adapthist

# class CadastralConfidenceEngine:
#     def __init__(self, target_size=500, local_radius_px=5.0):
#         self.target_size = target_size
#         self.local_radius = local_radius_px

#     def _evaluate_side_dissimilarity(self, pt, tangent_vec, gray_patch, window_size=5):
#         """
#         Samples small pixel windows on the absolute left and absolute right sides 
#         of a segment using its perpendicular normal vector. 
#         Returns a scaling multiplier based on side-to-side statistical distance.
#         """
#         normal_vec = np.array([-tangent_vec[1], tangent_vec[0]])
#         offset_distance = 3.0
        
#         left_center = pt + normal_vec * offset_distance
#         right_center = pt - normal_vec * offset_distance
        
#         left_pixels = []
#         right_pixels = []
#         half_w = window_size // 2
        
#         for dx in range(-half_w, half_w + 1):
#             for dy in range(-half_w, half_w + 1):
#                 lx, ly = int(np.round(left_center[0] + dx)), int(np.round(left_center[1] + dy))
#                 rx, ry = int(np.round(right_center[0] + dx)), int(np.round(right_center[1] + dy))
                
#                 if 0 <= lx < self.target_size and 0 <= ly < self.target_size:
#                     left_pixels.append(gray_patch[ly, lx])
#                 if 0 <= rx < self.target_size and 0 <= ry < self.target_size:
#                     right_pixels.append(gray_patch[ry, rx])
                    
#         if len(left_pixels) == 0 or len(right_pixels) == 0:
#             return 0.5
            
#         mean_left = np.mean(left_pixels)
#         mean_right = np.mean(right_pixels)
#         absolute_mean_diff = abs(mean_left - mean_right)
        
#         # Scale differences into an optimization reward
#         contrast_reward = 1.0 - np.exp(-5.0 * absolute_mean_diff)
#         return max(0.15, contrast_reward)
        
#     def _extract_confidence_field(self, gray_img):
#         """Applies local adaptive contrast scaling and structural ridge filtering."""
#         clahe_img = equalize_adapthist(gray_img, kernel_size=32, clip_limit=0.02)
#         ridge_response = meijering(clahe_img, sigmas=[1, 2], black_ridges=False)
        
#         if ridge_response.max() > 0:
#             ridge_response = (ridge_response - ridge_response.min()) / (ridge_response.max() - ridge_response.min())
#         return ridge_response

#     def _analyze_topology(self, skeleton_pts, ridge_map, gray_patch):
#         """Analyzes skeletonized paths to calculate direction vectors and assign scores."""
#         spatial_tree = KDTree(skeleton_pts)
#         neighbor_indices_list = spatial_tree.query_ball_point(skeleton_pts, r=self.local_radius)
        
#         classified_features = {}
#         tangent_map = {}         
#         pixel_confidence = {}    
        
#         for idx, pt in enumerate(skeleton_pts):
#             pt_tuple = (pt[0], pt[1])
#             neighbors = skeleton_pts[neighbor_indices_list[idx]]
#             density = len(neighbors)
            
#             if density < 3 or density > 18:
#                 continue
                
#             centroid = np.mean(neighbors, axis=0)
#             dist_to_centroid = np.linalg.norm(pt - centroid)
            
#             cov = np.cov(neighbors - centroid, rowvar=False)
#             if cov.shape != (2, 2) or np.any(np.isnan(cov)):
#                 continue
                
#             eigenvalues, eigenvectors = np.linalg.eigh(cov)
#             lam_min, lam_max = eigenvalues[0], eigenvalues[1]
#             local_ridge_strength = ridge_map[pt[1], pt[0]]
            
#             if lam_max > 0:
#                 linear_ratio = lam_min / lam_max
#                 tangent_vec = eigenvectors[:, 1]
                
#                 # Compute statistical profile divergence across normal vector
#                 side_contrast_multiplier = self._evaluate_side_dissimilarity(pt, tangent_vec, gray_patch)
                
#                 if linear_ratio <= 0.12 and dist_to_centroid <= 1.5:
#                     classified_features[pt_tuple] = 'line'
#                     tangent_map[pt_tuple] = tangent_vec
#                     base_score = 0.4 + (0.6 * local_ridge_strength * (1.0 - linear_ratio))
#                     pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier
                    
#                 elif linear_ratio <= 0.06 and dist_to_centroid > 1.5:
#                     classified_features[pt_tuple] = 'line'
#                     tangent_map[pt_tuple] = tangent_vec
#                     base_score = 0.3 + (0.7 * local_ridge_strength)
#                     pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier
                    
#                 elif linear_ratio > 0.15 and density >= 6 and dist_to_centroid <= 2.2:
#                     classified_features[pt_tuple] = 'junction'
#                     tangent_map[pt_tuple] = tangent_vec
#                     base_score = 0.5 + (0.5 * local_ridge_strength)
#                     pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier

#         return classified_features, tangent_map, pixel_confidence

#     def _stitch_and_generate_confidence_map(self, classified_features, tangent_map, pixel_confidence, ridge_map):
#         """Bridges tracing gaps along collinear trajectories with tight threshold gates."""
#         valid_pts = np.array(list(tangent_map.keys()))
#         confidence_map = np.zeros_like(ridge_map, dtype=float)
        
#         if len(valid_pts) == 0:
#             return confidence_map
            
#         feature_tree = KDTree(valid_pts)
#         for pt_tuple, conf in pixel_confidence.items():
#             confidence_map[pt_tuple[1], pt_tuple[0]] = conf

#         # Tightened configuration parameters to limit cross-field bleed
#         max_lookahead_dist = 10
#         parallel_exclusion_radius = 3.5
        
#         for i, pt in enumerate(valid_pts):
#             pt_tuple = (pt[0], pt[1])
#             pt_tangent = tangent_map[pt_tuple]
#             pt_type = classified_features[pt_tuple]
            
#             nearby_idx = feature_tree.query_ball_point(pt, r=max_lookahead_dist)
#             for n_idx in nearby_idx:
#                 target_pt = valid_pts[n_idx]
#                 target_tuple = (target_pt[0], target_pt[1])
                
#                 if np.array_equal(pt, target_pt):
#                     continue
                    
#                 target_tangent = tangent_map[target_tuple]
#                 displacement = target_pt - pt
#                 gap_dist = np.linalg.norm(displacement)
                
#                 if gap_dist == 0:
#                     continue
                    
#                 norm_displacement = displacement / gap_dist
#                 tangent_similarity = abs(np.dot(pt_tangent, target_tangent))
#                 collinearity = abs(np.dot(pt_tangent, norm_displacement))
                
#                 is_cross_merging = collinearity < 0.25 and tangent_similarity > 0.85 and gap_dist <= parallel_exclusion_radius
#                 is_valid_trajectory = (tangent_similarity > 0.92 and collinearity > 0.90) or (pt_type == 'junction' and tangent_similarity > 0.75)
                
#                 if is_valid_trajectory and not is_cross_merging:
#                     steps = int(np.ceil(gap_dist)) * 2
#                     path_pixels = []
#                     valid_step = True
                    
#                     for step in range(1, steps):
#                         interp_pt = pt + (displacement * (step / steps))
#                         ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
#                         if 0 <= ix < self.target_size and 0 <= iy < self.target_size:
#                             path_pixels.append(ridge_map[iy, ix])
#                         else:
#                             valid_step = False
#                             break
                            
#                     # UPDATED: Enforces high average pixel response across the bridge trajectory
#                     if valid_step and len(path_pixels) > 0 and np.mean(path_pixels) > 0.35:
#                         bridge_confidence = (pixel_confidence[pt_tuple] + pixel_confidence[target_tuple]) * 0.5 * tangent_similarity
                        
#                         for step in range(steps + 1):
#                             interp_pt = pt + (displacement * (step / steps))
#                             ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                            
#                             for dx in [-1, 0, 1]:
#                                 for dy in [-1, 0, 1]:
#                                     cx, cy = ix + dx, iy + dy
#                                     if 0 <= cx < self.target_size and 0 <= cy < self.target_size:
#                                         confidence_map[cy, cx] = max(confidence_map[cy, cx], bridge_confidence)
                                        
#         return confidence_map

#     def _apply_topological_length_sieve(self, confidence_map, min_line_length=18):
#         """
#         Identifies continuous components. 
#         Boosts long lines to 1.0 and strictly DROPS short lines to 0.0.
#         """
#         base_mask = confidence_map > 0.05
#         labeled_mask, num_features = ndimage.label(base_mask, structure=np.ones((3, 3)))
#         boosted_confidence = np.copy(confidence_map)
        
#         for feature_id in range(1, num_features + 1):
#             pixel_y, pixel_x = np.where(labeled_mask == feature_id)
#             if len(pixel_x) == 0:
#                 continue
                
#             delta_x = np.max(pixel_x) - np.min(pixel_x)
#             delta_y = np.max(pixel_y) - np.min(pixel_y)
#             line_extent = max(delta_x, delta_y)
            
#             # --- THE CORRECTED FILTER GATE ---
#             if line_extent >= min_line_length:
#                 boosted_confidence[pixel_y, pixel_x] = 1.0
#             else:
#                 # CRITICAL: This explicitly forces small fragments to disappear
#                 boosted_confidence[pixel_y, pixel_x] = 0.0
                
#         return boosted_confidence

#     def process_patch(self, gray_patch, min_line_length=18):
#         """Main execution sequence with diagnostics."""
        
#         gray_patch = gray_patch[:self.target_size, :self.target_size]

#         ridge_map = self._extract_confidence_field(gray_patch)

#         candidate_mask = ridge_map > 0.22
#         plt.figure(figsize=(8,8))
#         plt.imshow(candidate_mask, cmap='gray')
#         plt.title("Candidate Mask")
#         plt.show()

#         print("\n========== STAGE 1 ==========")
#         print(f"Candidate pixels: {np.count_nonzero(candidate_mask)}")

#         skeleton = skeletonize(candidate_mask)

#         skel_rows, skel_cols = np.where(skeleton)
#         skeleton_pts = np.column_stack((skel_cols, skel_rows))

#         print("\n========== STAGE 2 ==========")
#         print(f"Skeleton pixels: {len(skeleton_pts)}")

#         if len(skeleton_pts) == 0:
#             print("No skeleton points found.")
#             return np.zeros_like(gray_patch)

#         classified_features, tangent_map, pixel_confidence = self._analyze_topology(
#             skeleton_pts,
#             ridge_map,
#             gray_patch
#         )

#         print("\n========== STAGE 3 ==========")
#         print(f"Accepted structural points: {len(tangent_map)}")
#         print(f"Junctions: {sum(v == 'junction' for v in classified_features.values())}")
#         print(f"Lines: {sum(v == 'line' for v in classified_features.values())}")

#         raw_confidence_map = self._stitch_and_generate_confidence_map(
#             classified_features,
#             tangent_map,
#             pixel_confidence,
#             ridge_map
#         )

#         print("\n========== STAGE 4 ==========")
#         print(f"Pixels after stitching: {np.count_nonzero(raw_confidence_map > 0.05)}")

#         # Diagnostics BEFORE sieve
#         pre_mask = raw_confidence_map > 0.05
#         pre_labels, pre_count = ndimage.label(pre_mask)

#         print(f"Connected components before sieve: {pre_count}")

#         for comp_id in range(1, min(pre_count + 1, 15)):
#             ys, xs = np.where(pre_labels == comp_id)

#             component_pixels = len(xs)

#             delta_x = xs.max() - xs.min()
#             delta_y = ys.max() - ys.min()

#             extent = max(delta_x, delta_y)

#             print(
#                 f"Component {comp_id:3d} | "
#                 f"Pixels={component_pixels:5d} | "
#                 f"Extent={extent:4d}"
#             )

#         final_confidence_map = self._apply_topological_length_sieve(
#             raw_confidence_map,
#             min_line_length=min_line_length
#         )

#         print("\n========== STAGE 5 ==========")

#         post_mask = final_confidence_map > 0.05
#         post_labels, post_count = ndimage.label(post_mask)

#         print(f"Connected components after sieve: {post_count}")
#         print(f"Remaining pixels: {np.count_nonzero(post_mask)}")

#         return final_confidence_map


# # =========================================================================
# # RUNNER WITH TUNABLE LENGTH SIEVE SWITCH (STANDALONE GLOBAL FUNCTION)
# # =========================================================================
# def execute_tuned_confidence_run_new(geojson_path, plot_no, tiff_imagery="imagery.tif", min_confidence=0.42, min_line_length=18):
#     """Runs extraction, labels individual connected components on the graph, and filters by length."""
#     target_size = 500
#     gdf = gpd.read_file(geojson_path)
#     target_row = gdf.iloc[[plot_no]].copy()
    
#     with rasterio.open(tiff_imagery) as src:
#         if target_row.crs != src.crs:
#             target_row = target_row.to_crs(src.crs)
#         raw_geom = target_row.geometry.values[0]
#         vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
        
#         centroid = vector_poly.centroid
#         hw, hh = (target_size / 2) * abs(src.transform.a), (target_size / 2) * abs(src.transform.e)
#         bbox = box(centroid.x - hw, centroid.y - hh, centroid.x + hw, centroid.y + hh)
#         cropped_raster, _ = rasterio.mask.mask(src, [bbox], crop=True)
        
#     if cropped_raster.shape[0] >= 3:
#         img_patch = np.moveaxis(cropped_raster[:3], 0, -1).astype(float)
#         gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114)
#     else:
#         gray_patch = cropped_raster[0].astype(float)
#         img_patch = np.stack([gray_patch, gray_patch, gray_patch], axis=-1)
        
#     img_patch = img_patch[:target_size, :target_size]
#     gray_patch = gray_patch[:target_size, :target_size]
    
#     if gray_patch.max() > gray_patch.min():
#         gray_patch = (gray_patch - gray_patch.min()) / (gray_patch.max() - gray_patch.min())
#     if img_patch.max() > 0:
#         img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min())

#     # 1. Generate Raw Structural Map Up To Stitching Stage
#     engine = CadastralConfidenceEngine(target_size=target_size)
#     gray_patch_clipped = gray_patch[:engine.target_size, :engine.target_size]
#     ridge_map = engine._extract_confidence_field(gray_patch_clipped)
    
#     # Tightened threshold gate pass matching stage 1
#     candidate_mask = ridge_map > 0.42  
#     skeleton = skeletonize(candidate_mask)
#     skel_rows, skel_cols = np.where(skeleton)
#     skeleton_pts = np.column_stack((skel_cols, skel_rows))
    
#     if len(skeleton_pts) == 0:
#         print("No structural features extracted.")
#         return
        
#     classified_features, tangent_map, pixel_confidence = engine._analyze_topology(skeleton_pts, ridge_map, gray_patch_clipped)
#     raw_confidence_map = engine._stitch_and_generate_confidence_map(
#         classified_features, tangent_map, pixel_confidence, ridge_map
#     )
    
#     # 2. Compute Connected Components BEFORE the sieve for Graph Labeling
#     base_mask = raw_confidence_map > 0.05
#     labeled_mask, num_features = ndimage.label(base_mask, structure=np.ones((3, 3)))
    
#     # 3. Execute the Sieve Filter pass
#     final_confidence_map = engine._apply_topological_length_sieve(
#         raw_confidence_map, min_line_length=min_line_length
#     )
    
#     filtered_boundary_mask = final_confidence_map >= min_confidence
#     overlay_display = np.copy(img_patch)
#     overlay_display[filtered_boundary_mask] = [0.0, 1.0, 0.0] 
    
#     # 4. Generate the Plot Windows
#     plt.close('all') 
#     fig, axes = plt.subplots(1, 3, figsize=(20, 7), clear=True)
    
#     # Subplot 1: Input Source
#     axes[0].imshow(img_patch)
#     axes[0].set_title("Original Satellite Imagery")
#     axes[0].axis('off')
    
#     # Subplot 2: Labeled Confidence Map Matrix (Debugging Target)
#     axes[1].imshow(raw_confidence_map, cmap='hot')
#     axes[1].set_title("Pre-Sieve Matrix (Numbered Components)")
#     axes[1].axis('off')
    
#     # Identify the largest component to avoid filling the graph with identical numbers
#     component_sizes = [np.sum(labeled_mask == fid) for fid in range(1, num_features + 1)]
#     largest_component_id = np.argmax(component_sizes) + 1 if component_sizes else -1

#     # Print Component Locations onto the Figure canvas
#     for feature_id in range(1, min(num_features + 1, 150)):  # Caps text rendering at 150 traces for performance
#         pixel_y, pixel_x = np.where(labeled_mask == feature_id)
#         if len(pixel_x) == 0:
#             continue
            
#         # Place label text at the geometric center of the component string
#         mean_x = int(np.mean(pixel_x))
#         mean_y = int(np.mean(pixel_y))
        
#         # Color code: Cyan for small components, Blue for the massive interconnected web
#         text_color = '#00FFFF' if feature_id != largest_component_id else '#3388FF'
#         font_weight = 'bold' if feature_id == largest_component_id else 'normal'
        
#         axes[1].text(
#             mean_x, mean_y, str(feature_id),
#             color=text_color, fontsize=7, weight=font_weight,
#             ha='center', va='center',
#             bbox=dict(boxstyle='square,pad=0.1', fc='black', ec='none', alpha=0.6)
#         )
    
#     # Subplot 3: Final Output Output Display
#     axes[2].imshow(overlay_display)
#     axes[2].set_title(f"Protected Boundaries\n(Length Sieve >= {min_line_length}px)")
#     axes[2].axis('off')
    
#     plt.tight_layout()
#     fig.canvas.draw()
#     plt.show(block=True)

#     plt.figure(figsize=(8,8))
#     plt.imshow(candidate_mask, cmap='gray')
#     plt.title("Candidate Mask")
#     plt.show()


# # =========================================================================
# # RUN SCRIPT ENTRY POINT
# # =========================================================================
# if __name__ == "__main__":
#     execute_tuned_confidence_run_new(
#         geojson_path="input.geojson", 
#         plot_no=302, 
#         tiff_imagery="imagery.tif", 
#         min_confidence=0.22,      
#         min_line_length=20        
#     )


# =====================================================================================================================================================================
# =====================================================================================================================================================================


# # ============================================================================================================================
import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt

# Core image processing and structural frameworks
from scipy.spatial import KDTree
import scipy.ndimage as ndimage
from skimage.filters import meijering
from skimage.morphology import skeletonize
from skimage.exposure import equalize_adapthist

class CadastralConfidenceEngine:
    def __init__(self, target_size=500, local_radius_px=5.0):
        self.target_size = target_size
        self.local_radius = local_radius_px

    def _evaluate_side_dissimilarity(self, pt, tangent_vec, gray_patch, window_size=5):
        """
        Samples small pixel windows on the absolute left and absolute right sides 
        of a segment using its perpendicular normal vector. 
        Returns a scaling multiplier based on side-to-side statistical distance.
        """
        normal_vec = np.array([-tangent_vec[1], tangent_vec[0]])
        offset_distance = 3.0
        
        left_center = pt + normal_vec * offset_distance
        right_center = pt - normal_vec * offset_distance
        
        left_pixels = []
        right_pixels = []
        half_w = window_size // 2
        
        for dx in range(-half_w, half_w + 1):
            for dy in range(-half_w, half_w + 1):
                lx, ly = int(np.round(left_center[0] + dx)), int(np.round(left_center[1] + dy))
                rx, ry = int(np.round(right_center[0] + dx)), int(np.round(right_center[1] + dy))
                
                if 0 <= lx < self.target_size and 0 <= ly < self.target_size:
                    left_pixels.append(gray_patch[ly, lx])
                if 0 <= rx < self.target_size and 0 <= ry < self.target_size:
                    right_pixels.append(gray_patch[ry, rx])
                    
        if len(left_pixels) == 0 or len(right_pixels) == 0:
            return 0.5
            
        mean_left = np.mean(left_pixels)
        mean_right = np.mean(right_pixels)
        absolute_mean_diff = abs(mean_left - mean_right)
        
        # Scale differences into an optimization reward
        contrast_reward = 1.0 - np.exp(-5.0 * absolute_mean_diff)
        return max(0.15, contrast_reward)
        
    def _extract_confidence_field(self, gray_img):
        """Applies local adaptive contrast scaling and structural ridge filtering."""
        clahe_img = equalize_adapthist(gray_img, kernel_size=32, clip_limit=0.02)
        ridge_response = meijering(clahe_img, sigmas=[1, 2], black_ridges=False)
        
        if ridge_response.max() > 0:
            ridge_response = (ridge_response - ridge_response.min()) / (ridge_response.max() - ridge_response.min())
        return ridge_response

    def _analyze_topology(self, skeleton_pts, ridge_map, gray_patch):
        """Analyzes skeletonized paths to calculate direction vectors and assign scores."""
        spatial_tree = KDTree(skeleton_pts)
        neighbor_indices_list = spatial_tree.query_ball_point(skeleton_pts, r=self.local_radius)
        
        classified_features = {}
        tangent_map = {}         
        pixel_confidence = {}    
        
        for idx, pt in enumerate(skeleton_pts):
            pt_tuple = (pt[0], pt[1])
            neighbors = skeleton_pts[neighbor_indices_list[idx]]
            density = len(neighbors)
            
            if density < 3 or density > 18:
                continue
                
            centroid = np.mean(neighbors, axis=0)
            dist_to_centroid = np.linalg.norm(pt - centroid)
            
            cov = np.cov(neighbors - centroid, rowvar=False)
            if cov.shape != (2, 2) or np.any(np.isnan(cov)):
                continue
                
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            lam_min, lam_max = eigenvalues[0], eigenvalues[1]
            local_ridge_strength = ridge_map[pt[1], pt[0]]
            
            if lam_max > 0:
                linear_ratio = lam_min / lam_max
                tangent_vec = eigenvectors[:, 1]
                
                # Compute statistical profile divergence across normal vector
                side_contrast_multiplier = self._evaluate_side_dissimilarity(pt, tangent_vec, gray_patch)
                
                if linear_ratio <= 0.12 and dist_to_centroid <= 1.5:
                    classified_features[pt_tuple] = 'line'
                    tangent_map[pt_tuple] = tangent_vec
                    base_score = 0.4 + (0.6 * local_ridge_strength * (1.0 - linear_ratio))
                    pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier
                    
                elif linear_ratio <= 0.06 and dist_to_centroid > 1.5:
                    classified_features[pt_tuple] = 'line'
                    tangent_map[pt_tuple] = tangent_vec
                    base_score = 0.3 + (0.7 * local_ridge_strength)
                    pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier
                    
                elif linear_ratio > 0.15 and density >= 6 and dist_to_centroid <= 2.2:
                    classified_features[pt_tuple] = 'junction'
                    tangent_map[pt_tuple] = tangent_vec
                    base_score = 0.5 + (0.5 * local_ridge_strength)
                    pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier

        return classified_features, tangent_map, pixel_confidence

    def _stitch_and_generate_confidence_map(self, classified_features, tangent_map, pixel_confidence, ridge_map):
        """Bridges tracing gaps along collinear trajectories with tight threshold gates."""
        valid_pts = np.array(list(tangent_map.keys()))
        confidence_map = np.zeros_like(ridge_map, dtype=float)
        
        if len(valid_pts) == 0:
            return confidence_map
            
        feature_tree = KDTree(valid_pts)
        for pt_tuple, conf in pixel_confidence.items():
            confidence_map[pt_tuple[1], pt_tuple[0]] = conf

        # Tightened configuration parameters to limit cross-field bleed
        max_lookahead_dist = 10
        parallel_exclusion_radius = 3.5
        
        for i, pt in enumerate(valid_pts):
            pt_tuple = (pt[0], pt[1])
            pt_tangent = tangent_map[pt_tuple]
            pt_type = classified_features[pt_tuple]
            
            nearby_idx = feature_tree.query_ball_point(pt, r=max_lookahead_dist)
            for n_idx in nearby_idx:
                target_pt = valid_pts[n_idx]
                target_tuple = (target_pt[0], target_pt[1])
                
                if np.array_equal(pt, target_pt):
                    continue
                    
                target_tangent = tangent_map[target_tuple]
                displacement = target_pt - pt
                gap_dist = np.linalg.norm(displacement)
                
                if gap_dist == 0:
                    continue
                    
                norm_displacement = displacement / gap_dist
                tangent_similarity = abs(np.dot(pt_tangent, target_tangent))
                collinearity = abs(np.dot(pt_tangent, norm_displacement))
                
                is_cross_merging = collinearity < 0.25 and tangent_similarity > 0.85 and gap_dist <= parallel_exclusion_radius
                is_valid_trajectory = (tangent_similarity > 0.92 and collinearity > 0.90) or (pt_type == 'junction' and tangent_similarity > 0.75)
                
                if is_valid_trajectory and not is_cross_merging:
                    steps = int(np.ceil(gap_dist)) * 2
                    path_pixels = []
                    valid_step = True
                    
                    for step in range(1, steps):
                        interp_pt = pt + (displacement * (step / steps))
                        ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                        if 0 <= ix < self.target_size and 0 <= iy < self.target_size:
                            path_pixels.append(ridge_map[iy, ix])
                        else:
                            valid_step = False
                            break
                            
                    # UPDATED: Enforces high average pixel response across the bridge trajectory
                    if valid_step and len(path_pixels) > 0 and np.mean(path_pixels) > 0.35:
                        bridge_confidence = (pixel_confidence[pt_tuple] + pixel_confidence[target_tuple]) * 0.5 * tangent_similarity
                        
                        for step in range(steps + 1):
                            interp_pt = pt + (displacement * (step / steps))
                            ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                            
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    cx, cy = ix + dx, iy + dy
                                    if 0 <= cx < self.target_size and 0 <= cy < self.target_size:
                                        confidence_map[cy, cx] = max(confidence_map[cy, cx], bridge_confidence)

        # # 1. This is your raw binary mask from Step 1 (the one from your plot)           
        return confidence_map
    

    
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import KDTree
    from skimage.morphology import skeletonize

    def run_sieve_then_stitch_pipeline(candidate_mask, ridge_map, min_pixel_clean=30, max_gap_dist=15.0):
        """
        Pure pipeline execution:
        1. Area Sieve on raw candidate pixels to remove salt-and-pepper noise.
        2. Skeletonization of clean lines only.
        3. Endpoint-based topological bridging over gaps.
        """
        target_size = candidate_mask.shape[0]
        
        # ==========================================
        # STEP 1: PIXEL SIEVE (PRE-CLEANING)
        # ==========================================
        # Label everything connected in the raw mask
        labeled_mask, num_features = ndimage.label(candidate_mask, structure=np.ones((3,3)))
        clean_candidate_mask = np.zeros_like(candidate_mask, dtype=bool)
        
        # Only keep pixel clusters that are large enough to be real segments
        for feature_id in range(1, num_features + 1):
            pixel_y, pixel_x = np.where(labeled_mask == feature_id)
            if len(pixel_x) >= min_pixel_clean:
                clean_candidate_mask[pixel_y, pixel_x] = True

        # ==========================================
        # STEP 2: SKELETONIZE THE CLEANED PATHS
        # ==========================================
        skeleton = skeletonize(clean_candidate_mask)
        
        # ==========================================
        # STEP 3: TERMINAL ENDPOINT EXTRACTION
        # ==========================================
        kernel = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0) * skeleton
        endpoints_coordinates = np.column_stack(np.where(neighbor_count == 1))[:, ::-1] # Convert to [x, y]
        
        # Estimate outward directions for endpoints
        ep_registry = []
        y_indices, x_indices = np.where(skeleton)
        skel_pts = np.column_stack((x_indices, y_indices))
        
        for ep in endpoints_coordinates:
            distances = np.linalg.norm(skel_pts - ep, axis=1)
            local_pts = skel_pts[distances <= 8]
            
            if len(local_pts) >= 3:
                centroid = np.mean(local_pts, axis=0)
                cov = np.cov(local_pts - centroid, rowvar=False)
                if cov.shape == (2, 2) and not np.any(np.isnan(cov)):
                    _, eigenvectors = np.linalg.eigh(cov)
                    tangent = eigenvectors[:, 1]
                    # Orient vector outward away from core body
                    if np.dot(tangent, ep - centroid) < 0:
                        tangent = -tangent
                    ep_registry.append({
                        'pos': ep, 
                        'tangent': tangent / (np.linalg.norm(tangent) + 1e-6), 
                        'consumed': False
                    })

        # ==========================================
        # STEP 4: ENDPOINT-TO-ENDPOINT STITCHING
        # ==========================================
        output_skeleton = np.copy(skeleton)
        candidate_pairs = []
        num_eps = len(ep_registry)
        
        for i in range(num_eps):
            for j in range(i + 1, num_eps):
                ep_A = ep_registry[i]
                ep_B = ep_registry[j]
                
                disp = ep_B['pos'] - ep_A['pos']
                dist = np.linalg.norm(disp)
                
                if dist > max_gap_dist or dist < 2.0:
                    continue
                    
                gap_vector = disp / dist
                tangent_sim = abs(np.dot(ep_A['tangent'], ep_B['tangent']))
                alignment_A = np.dot(ep_A['tangent'], gap_vector)
                alignment_B = np.dot(ep_B['tangent'], -gap_vector)
                
                # Straight-line continuity match criteria
                if tangent_sim >= 0.90 and alignment_A >= 0.90 and alignment_B >= 0.90:
                    candidate_pairs.append({
                        'idx_A': i, 'idx_B': j, 'dist': dist,
                        'pos_A': ep_A['pos'], 'pos_B': ep_B['pos']
                    })
                    
        # Connect closer gaps first
        candidate_pairs = sorted(candidate_pairs, key=lambda x: x['dist'])
        
        for pair in candidate_pairs:
            if ep_registry[pair['idx_A']]['consumed'] or ep_registry[pair['idx_B']]['consumed']:
                continue
                
            # Verify ridge background verification gate
            steps = int(np.ceil(pair['dist'])) * 2
            path_pixels = []
            for step in range(1, steps):
                interp = pair['pos_A'] + (pair['pos_B'] - pair['pos_A']) * (step / steps)
                ix, iy = int(np.round(interp[0])), int(np.round(interp[1]))
                if 0 <= ix < target_size and 0 <= iy < target_size:
                    path_pixels.append(ridge_map[iy, ix])
                    
            if len(path_pixels) > 0 and np.mean(path_pixels) >= 0.25:
                ep_registry[pair['idx_A']]['consumed'] = True
                ep_registry[pair['idx_B']]['consumed'] = True
                
                # Generate 1-pixel line bridge entries
                for step in range(steps + 1):
                    interp = pair['pos_A'] + (pair['pos_B'] - pair['pos_A']) * (step / steps)
                    ix, iy = int(np.round(interp[0])), int(np.round(interp[1]))
                    if 0 <= ix < target_size and 0 <= iy < target_size:
                        output_skeleton[iy, ix] = True

                return clean_candidate_mask, output_skeleton




    def _apply_topological_length_sieve(self, confidence_map, min_line_length=18):
        """
        Identifies continuous components. 
        Boosts long lines to 1.0 and strictly DROPS short lines to 0.0.
        """
        base_mask = confidence_map > 0.05
        labeled_mask, num_features = ndimage.label(base_mask, structure=np.ones((3, 3)))
        boosted_confidence = np.copy(confidence_map)
        
        for feature_id in range(1, num_features + 1):
            pixel_y, pixel_x = np.where(labeled_mask == feature_id)
            if len(pixel_x) == 0:
                continue
                
            delta_x = np.max(pixel_x) - np.min(pixel_x)
            delta_y = np.max(pixel_y) - np.min(pixel_y)
            line_extent = max(delta_x, delta_y)
            
            # --- THE CORRECTED FILTER GATE ---
            if line_extent >= min_line_length:
                boosted_confidence[pixel_y, pixel_x] = 1.0
            else:
                # CRITICAL: This explicitly forces small fragments to disappear
                boosted_confidence[pixel_y, pixel_x] = 0.0
                
        return boosted_confidence

    def process_patch(self, gray_patch, min_line_length=18):
        """Main execution sequence with diagnostics."""
        
        gray_patch = gray_patch[:self.target_size, :self.target_size]

        ridge_map = self._extract_confidence_field(gray_patch)

        candidate_mask = ridge_map > 0.22
        plt.figure(figsize=(8,8))
        plt.imshow(candidate_mask, cmap='gray')
        plt.title("Candidate Mask")
        plt.show()

        print("\n========== STAGE 1 ==========")
        print(f"Candidate pixels: {np.count_nonzero(candidate_mask)}")

        skeleton = skeletonize(candidate_mask)

        skel_rows, skel_cols = np.where(skeleton)
        skeleton_pts = np.column_stack((skel_cols, skel_rows))

        print("\n========== STAGE 2 ==========")
        print(f"Skeleton pixels: {len(skeleton_pts)}")

        if len(skeleton_pts) == 0:
            print("No skeleton points found.")
            return np.zeros_like(gray_patch)

        classified_features, tangent_map, pixel_confidence = self._analyze_topology(
            skeleton_pts,
            ridge_map,
            gray_patch
        )

        print("\n========== STAGE 3 ==========")
        print(f"Accepted structural points: {len(tangent_map)}")
        print(f"Junctions: {sum(v == 'junction' for v in classified_features.values())}")
        print(f"Lines: {sum(v == 'line' for v in classified_features.values())}")

        raw_confidence_map = self._stitch_and_generate_confidence_map(
            classified_features,
            tangent_map,
            pixel_confidence,
            ridge_map
        )

        print("\n========== STAGE 4 ==========")
        print(f"Pixels after stitching: {np.count_nonzero(raw_confidence_map > 0.05)}")

        # Diagnostics BEFORE sieve
        pre_mask = raw_confidence_map > 0.05
        pre_labels, pre_count = ndimage.label(pre_mask)

        print(f"Connected components before sieve: {pre_count}")

        for comp_id in range(1, min(pre_count + 1, 15)):
            ys, xs = np.where(pre_labels == comp_id)

            component_pixels = len(xs)

            delta_x = xs.max() - xs.min()
            delta_y = ys.max() - ys.min()

            extent = max(delta_x, delta_y)

            print(
                f"Component {comp_id:3d} | "
                f"Pixels={component_pixels:5d} | "
                f"Extent={extent:4d}"
            )

        final_confidence_map = self._apply_topological_length_sieve(
            raw_confidence_map,
            min_line_length=min_line_length
        )

        print("\n========== STAGE 5 ==========")

        post_mask = final_confidence_map > 0.05
        post_labels, post_count = ndimage.label(post_mask)

        print(f"Connected components after sieve: {post_count}")
        print(f"Remaining pixels: {np.count_nonzero(post_mask)}")

        return final_confidence_map
    



# =========================================================================
# RUNNER WITH TUNABLE LENGTH SIEVE SWITCH (STANDALONE GLOBAL FUNCTION)
# =========================================================================
def execute_tuned_confidence_run_new(geojson_path, plot_no, tiff_imagery="imagery.tif", min_confidence=0.42, min_line_length=18):
    """Runs extraction, labels individual connected components on the graph, and filters by length."""
    target_size = 500
    gdf = gpd.read_file(geojson_path)
    target_row = gdf.iloc[[plot_no]].copy()
    
    with rasterio.open(tiff_imagery) as src:
        if target_row.crs != src.crs:
            target_row = target_row.to_crs(src.crs)
        raw_geom = target_row.geometry.values[0]
        vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
        
        centroid = vector_poly.centroid
        hw, hh = (target_size / 2) * abs(src.transform.a), (target_size / 2) * abs(src.transform.e)
        bbox = box(centroid.x - hw, centroid.y - hh, centroid.x + hw, centroid.y + hh)
        cropped_raster, _ = rasterio.mask.mask(src, [bbox], crop=True)
        
    if cropped_raster.shape[0] >= 3:
        img_patch = np.moveaxis(cropped_raster[:3], 0, -1).astype(float)
        gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114)
    else:
        gray_patch = cropped_raster[0].astype(float)
        img_patch = np.stack([gray_patch, gray_patch, gray_patch], axis=-1)
        
    img_patch = img_patch[:target_size, :target_size]
    gray_patch = gray_patch[:target_size, :target_size]
    
    if gray_patch.max() > gray_patch.min():
        gray_patch = (gray_patch - gray_patch.min()) / (gray_patch.max() - gray_patch.min())
    if img_patch.max() > 0:
        img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min())

    # 1. Generate Raw Structural Map Up To Stitching Stage
    engine = CadastralConfidenceEngine(target_size=target_size)
    gray_patch_clipped = gray_patch[:engine.target_size, :engine.target_size]
    ridge_map = engine._extract_confidence_field(gray_patch_clipped)
    
    # Tightened threshold gate pass matching stage 1
    candidate_mask = ridge_map > 0.42  
    skeleton = skeletonize(candidate_mask)
    skel_rows, skel_cols = np.where(skeleton)
    skeleton_pts = np.column_stack((skel_cols, skel_rows))
    
    if len(skeleton_pts) == 0:
        print("No structural features extracted.")
        return
        
    classified_features, tangent_map, pixel_confidence = engine._analyze_topology(skeleton_pts, ridge_map, gray_patch_clipped)
    raw_confidence_map = engine._stitch_and_generate_confidence_map(
        classified_features, tangent_map, pixel_confidence, ridge_map
    )
    
    # 2. Compute Connected Components BEFORE the sieve for Graph Labeling
    base_mask = raw_confidence_map > 0.05
    labeled_mask, num_features = ndimage.label(base_mask, structure=np.ones((3, 3)))
    
    # 3. Execute the Sieve Filter pass
    final_confidence_map = engine._apply_topological_length_sieve(
        raw_confidence_map, min_line_length=min_line_length
    )
    
    filtered_boundary_mask = final_confidence_map >= min_confidence
    overlay_display = np.copy(img_patch)
    overlay_display[filtered_boundary_mask] = [0.0, 1.0, 0.0] 
    
    # 4. Generate the Plot Windows
    plt.close('all') 
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), clear=True)
    
    # Subplot 1: Input Source
    axes[0].imshow(img_patch)
    axes[0].set_title("Original Satellite Imagery")
    axes[0].axis('off')
    
    # Subplot 2: Labeled Confidence Map Matrix (Debugging Target)
    axes[1].imshow(raw_confidence_map, cmap='hot')
    axes[1].set_title("Pre-Sieve Matrix (Numbered Components)")
    axes[1].axis('off')
    
    # Identify the largest component to avoid filling the graph with identical numbers
    component_sizes = [np.sum(labeled_mask == fid) for fid in range(1, num_features + 1)]
    largest_component_id = np.argmax(component_sizes) + 1 if component_sizes else -1

    # Print Component Locations onto the Figure canvas
    for feature_id in range(1, min(num_features + 1, 150)):  # Caps text rendering at 150 traces for performance
        pixel_y, pixel_x = np.where(labeled_mask == feature_id)
        if len(pixel_x) == 0:
            continue
            
        # Place label text at the geometric center of the component string
        mean_x = int(np.mean(pixel_x))
        mean_y = int(np.mean(pixel_y))
        
        # Color code: Cyan for small components, Blue for the massive interconnected web
        text_color = '#00FFFF' if feature_id != largest_component_id else '#3388FF'
        font_weight = 'bold' if feature_id == largest_component_id else 'normal'
        
        axes[1].text(
            mean_x, mean_y, str(feature_id),
            color=text_color, fontsize=7, weight=font_weight,
            ha='center', va='center',
            bbox=dict(boxstyle='square,pad=0.1', fc='black', ec='none', alpha=0.6)
        )
    
    # Subplot 3: Final Output Output Display
    axes[2].imshow(overlay_display)
    axes[2].set_title(f"Protected Boundaries\n(Length Sieve >= {min_line_length}px)")
    axes[2].axis('off')
    
    plt.tight_layout()
    fig.canvas.draw()
    plt.show(block=True)

    plt.figure(figsize=(8,8))
    plt.imshow(candidate_mask, cmap='gray')
    plt.title("Candidate Mask")
    plt.show()


# =========================================================================
# RUN SCRIPT ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    execute_tuned_confidence_run_new(
        geojson_path="input.geojson", 
        plot_no=300, 
        tiff_imagery="imagery.tif", 
        min_confidence=0.22,      
        min_line_length=30        
    )

# ================================================================================================================================================================



# ============================================================================================================================
import os
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import box, MultiPolygon
import matplotlib.pyplot as plt

# Core image processing and structural frameworks
from scipy.spatial import KDTree
import scipy.ndimage as ndimage
from skimage.filters import meijering
from skimage.morphology import skeletonize
from skimage.exposure import equalize_adapthist

# =========================================================================
# STANDALONE PIPELINE: PIXEL SIEVE & ENDPOINT STITCHING
# =========================================================================
def run_sieve_then_stitch_pipeline(candidate_mask, ridge_map, min_pixel_clean=30, max_gap_dist=15.0):
    """
    Pure pipeline execution:
    1. Area Sieve on raw candidate pixels to remove salt-and-pepper noise.
    2. Skeletonization of clean lines only.
    3. Endpoint-based topological bridging over gaps.
    """
    target_size = candidate_mask.shape[0]
    
    # ==========================================
    # STEP 1: PIXEL SIEVE (PRE-CLEANING)
    # ==========================================
    # Label everything connected in the raw mask
    labeled_mask, num_features = ndimage.label(candidate_mask, structure=np.ones((3,3)))
    clean_candidate_mask = np.zeros_like(candidate_mask, dtype=bool)
    
    # Only keep pixel clusters that are large enough to be real segments
    for feature_id in range(1, num_features + 1):
        pixel_y, pixel_x = np.where(labeled_mask == feature_id)
        if len(pixel_x) >= min_pixel_clean:
            clean_candidate_mask[pixel_y, pixel_x] = True

    # ==========================================
    # STEP 2: SKELETONIZE THE CLEANED PATHS
    # ==========================================
    skeleton = skeletonize(clean_candidate_mask)
    
    # ==========================================
    # STEP 3: TERMINAL ENDPOINT EXTRACTION
    # ==========================================
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0) * skeleton
    endpoints_coordinates = np.column_stack(np.where(neighbor_count == 1))[:, ::-1] # Convert to [x, y]
    
    # Estimate outward directions for endpoints
    ep_registry = []
    y_indices, x_indices = np.where(skeleton)
    skel_pts = np.column_stack((x_indices, y_indices))
    
    for ep in endpoints_coordinates:
        distances = np.linalg.norm(skel_pts - ep, axis=1)
        local_pts = skel_pts[distances <= 8]
        
        if len(local_pts) >= 3:
            centroid = np.mean(local_pts, axis=0)
            cov = np.cov(local_pts - centroid, rowvar=False)
            if cov.shape == (2, 2) and not np.any(np.isnan(cov)):
                _, eigenvectors = np.linalg.eigh(cov)
                tangent = eigenvectors[:, 1]
                # Orient vector outward away from core body
                if np.dot(tangent, ep - centroid) < 0:
                    tangent = -tangent
                ep_registry.append({
                    'pos': ep, 
                    'tangent': tangent / (np.linalg.norm(tangent) + 1e-6), 
                    'consumed': False
                })

    # ==========================================
    # STEP 4: ENDPOINT-TO-ENDPOINT STITCHING
    # ==========================================
    output_skeleton = np.copy(skeleton)
    candidate_pairs = []
    num_eps = len(ep_registry)
    
    for i in range(num_eps):
        for j in range(i + 1, num_eps):
            ep_A = ep_registry[i]
            ep_B = ep_registry[j]
            
            disp = ep_B['pos'] - ep_A['pos']
            dist = np.linalg.norm(disp)
            
            if dist > max_gap_dist or dist < 2.0:
                continue
                
            gap_vector = disp / dist
            tangent_sim = abs(np.dot(ep_A['tangent'], ep_B['tangent']))
            alignment_A = np.dot(ep_A['tangent'], gap_vector)
            alignment_B = np.dot(ep_B['tangent'], -gap_vector)
            
            # Straight-line continuity match criteria
            if tangent_sim >= 0.90 and alignment_A >= 0.90 and alignment_B >= 0.90:
                candidate_pairs.append({
                    'idx_A': i, 'idx_B': j, 'dist': dist,
                    'pos_A': ep_A['pos'], 'pos_B': ep_B['pos']
                })
                
    # Connect closer gaps first
    candidate_pairs = sorted(candidate_pairs, key=lambda x: x['dist'])
    
    for pair in candidate_pairs:
        if ep_registry[pair['idx_A']]['consumed'] or ep_registry[pair['idx_B']]['consumed']:
            continue
            
        # Verify ridge background verification gate
        steps = int(np.ceil(pair['dist'])) * 2
        path_pixels = []
        for step in range(1, steps):
            interp = pair['pos_A'] + (pair['pos_B'] - pair['pos_A']) * (step / steps)
            ix, iy = int(np.round(interp[0])), int(np.round(interp[1]))
            if 0 <= ix < target_size and 0 <= iy < target_size:
                path_pixels.append(ridge_map[iy, ix])
                
        if len(path_pixels) > 0 and np.mean(path_pixels) >= 0.25:
            ep_registry[pair['idx_A']]['consumed'] = True
            ep_registry[pair['idx_B']]['consumed'] = True
            
            # Generate 1-pixel line bridge entries
            for step in range(steps + 1):
                interp = pair['pos_A'] + (pair['pos_B'] - pair['pos_A']) * (step / steps)
                ix, iy = int(np.round(interp[0])), int(np.round(interp[1]))
                if 0 <= ix < target_size and 0 <= iy < target_size:
                    output_skeleton[iy, ix] = True

    return clean_candidate_mask, output_skeleton


class CadastralConfidenceEngine:
    def __init__(self, target_size=5000, local_radius_px=5.0):
        self.target_size = target_size
        self.local_radius = local_radius_px

    def _evaluate_side_dissimilarity(self, pt, tangent_vec, gray_patch, window_size=5):
        """
        Samples small pixel windows on the absolute left and absolute right sides 
        of a segment using its perpendicular normal vector. 
        Returns a scaling multiplier based on side-to-side statistical distance.
        """
        normal_vec = np.array([-tangent_vec[1], tangent_vec[0]])
        offset_distance = 3.0
        
        left_center = pt + normal_vec * offset_distance
        right_center = pt - normal_vec * offset_distance
        
        left_pixels = []
        right_pixels = []
        half_w = window_size // 2
        
        for dx in range(-half_w, half_w + 1):
            for dy in range(-half_w, half_w + 1):
                lx, ly = int(np.round(left_center[0] + dx)), int(np.round(left_center[1] + dy))
                rx, ry = int(np.round(right_center[0] + dx)), int(np.round(right_center[1] + dy))
                
                if 0 <= lx < self.target_size and 0 <= ly < self.target_size:
                    left_pixels.append(gray_patch[ly, lx])
                if 0 <= rx < self.target_size and 0 <= ry < self.target_size:
                    right_pixels.append(gray_patch[ry, rx])
                    
        if len(left_pixels) == 0 or len(right_pixels) == 0:
            return 0.5
            
        mean_left = np.mean(left_pixels)
        mean_right = np.mean(right_pixels)
        absolute_mean_diff = abs(mean_left - mean_right)
        
        # Scale differences into an optimization reward
        contrast_reward = 1.0 - np.exp(-5.0 * absolute_mean_diff)
        return max(0.15, contrast_reward)
        
    def _extract_confidence_field(self, gray_img):
        """Applies local adaptive contrast scaling and structural ridge filtering."""
        clahe_img = equalize_adapthist(gray_img, kernel_size=32, clip_limit=0.02)
        ridge_response = meijering(clahe_img, sigmas=[1, 2], black_ridges=False)
        
        if ridge_response.max() > 0:
            ridge_response = (ridge_response - ridge_response.min()) / (ridge_response.max() - ridge_response.min())
        return ridge_response

    def _analyze_topology(self, skeleton_pts, ridge_map, gray_patch):
        """Analyzes skeletonized paths to calculate direction vectors and assign scores."""
        spatial_tree = KDTree(skeleton_pts)
        neighbor_indices_list = spatial_tree.query_ball_point(skeleton_pts, r=self.local_radius)
        
        classified_features = {}
        tangent_map = {}         
        pixel_confidence = {}    
        
        for idx, pt in enumerate(skeleton_pts):
            pt_tuple = (pt[0], pt[1])
            neighbors = skeleton_pts[neighbor_indices_list[idx]]
            density = len(neighbors)
            
            if density < 3 or density > 18:
                continue
                
            centroid = np.mean(neighbors, axis=0)
            dist_to_centroid = np.linalg.norm(pt - centroid)
            
            cov = np.cov(neighbors - centroid, rowvar=False)
            if cov.shape != (2, 2) or np.any(np.isnan(cov)):
                continue
                
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            lam_min, lam_max = eigenvalues[0], eigenvalues[1]
            local_ridge_strength = ridge_map[pt[1], pt[0]]
            
            if lam_max > 0:
                linear_ratio = lam_min / lam_max
                tangent_vec = eigenvectors[:, 1]
                
                # Compute statistical profile divergence across normal vector
                side_contrast_multiplier = self._evaluate_side_dissimilarity(pt, tangent_vec, gray_patch)
                
                if linear_ratio <= 0.12 and dist_to_centroid <= 1.5:
                    classified_features[pt_tuple] = 'line'
                    tangent_map[pt_tuple] = tangent_vec
                    base_score = 0.4 + (0.6 * local_ridge_strength * (1.0 - linear_ratio))
                    pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier
                    
                elif linear_ratio <= 0.06 and dist_to_centroid > 1.5:
                    classified_features[pt_tuple] = 'line'
                    tangent_map[pt_tuple] = tangent_vec
                    base_score = 0.3 + (0.7 * local_ridge_strength)
                    pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier
                    
                elif linear_ratio > 0.15 and density >= 6 and dist_to_centroid <= 2.2:
                    classified_features[pt_tuple] = 'junction'
                    tangent_map[pt_tuple] = tangent_vec
                    base_score = 0.5 + (0.5 * local_ridge_strength)
                    pixel_confidence[pt_tuple] = base_score * side_contrast_multiplier

        return classified_features, tangent_map, pixel_confidence

    def _stitch_and_generate_confidence_map(self, classified_features, tangent_map, pixel_confidence, ridge_map):
        """Bridges tracing gaps along collinear trajectories with tight threshold gates."""
        valid_pts = np.array(list(tangent_map.keys()))
        confidence_map = np.zeros_like(ridge_map, dtype=float)
        
        if len(valid_pts) == 0:
            return candidate_mask, candidate_mask, confidence_map
            
        feature_tree = KDTree(valid_pts)
        for pt_tuple, conf in pixel_confidence.items():
            confidence_map[pt_tuple[1], pt_tuple[0]] = conf

        # Tightened configuration parameters to limit cross-field bleed
        max_lookahead_dist = 10
        parallel_exclusion_radius = 3.5
        
        for i, pt in enumerate(valid_pts):
            pt_tuple = (pt[0], pt[1])
            pt_tangent = tangent_map[pt_tuple]
            pt_type = classified_features[pt_tuple]
            
            nearby_idx = feature_tree.query_ball_point(pt, r=max_lookahead_dist)
            for n_idx in nearby_idx:
                target_pt = valid_pts[n_idx]
                target_tuple = (target_pt[0], target_pt[1])
                
                if np.array_equal(pt, target_pt):
                    continue
                    
                target_tangent = tangent_map[target_tuple]
                displacement = target_pt - pt
                gap_dist = np.linalg.norm(displacement)
                
                if gap_dist == 0:
                    continue
                    
                norm_displacement = displacement / gap_dist
                tangent_similarity = abs(np.dot(pt_tangent, target_tangent))
                collinearity = abs(np.dot(pt_tangent, norm_displacement))
                
                is_cross_merging = collinearity < 0.25 and tangent_similarity > 0.85 and gap_dist <= parallel_exclusion_radius
                is_valid_trajectory = (tangent_similarity > 0.92 and collinearity > 0.90) or (pt_type == 'junction' and tangent_similarity > 0.75)
                
                if is_valid_trajectory and not is_cross_merging:
                    steps = int(np.ceil(gap_dist)) * 2
                    path_pixels = []
                    valid_step = True
                    
                    for step in range(1, steps):
                        interp_pt = pt + (displacement * (step / steps))
                        ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                        if 0 <= ix < self.target_size and 0 <= iy < self.target_size:
                            path_pixels.append(ridge_map[iy, ix])
                        else:
                            valid_step = False
                            break
                            
                    # UPDATED: Enforces high average pixel response across the bridge trajectory
                    if valid_step and len(path_pixels) > 0 and np.mean(path_pixels) > 0.35:
                        bridge_confidence = (pixel_confidence[pt_tuple] + pixel_confidence[target_tuple]) * 0.5 * tangent_similarity
                        
                        for step in range(steps + 1):
                            interp_pt = pt + (displacement * (step / steps))
                            ix, iy = int(np.round(interp_pt[0])), int(np.round(interp_pt[1]))
                            
                            for dx in [-1, 0, 1]:
                                for dy in [-1, 0, 1]:
                                    cx, cy = ix + dx, iy + dy
                                    if 0 <= cx < self.target_size and 0 <= cy < self.target_size:
                                        confidence_map[cy, cx] = max(confidence_map[cy, cx], bridge_confidence)

        # 1. This is your raw binary mask from Step 1 (the one from your plot)
        candidate_mask = ridge_map > 0.42  

        # 2. Run the new clean pipeline
        # min_pixel_clean=30 drops any noise cluster smaller than 30 pixels total
        clean_mask, final_skeleton = run_sieve_then_stitch_pipeline(
            candidate_mask=candidate_mask, 
            ridge_map=ridge_map, 
            min_pixel_clean=30, 
            max_gap_dist=15.0
        )
                                                
        return candidate_mask, clean_mask, final_skeleton

    def _apply_topological_length_sieve(self, confidence_map, min_line_length=18):
        """
        Identifies continuous components. 
        Boosts long lines to 1.0 and strictly DROPS short lines to 0.0.
        """
        base_mask = confidence_map > 0.05
        labeled_mask, num_features = ndimage.label(base_mask, structure=np.ones((3, 3)))
        boosted_confidence = np.copy(confidence_map)
        
        for feature_id in range(1, num_features + 1):
            pixel_y, pixel_x = np.where(labeled_mask == feature_id)
            if len(pixel_x) == 0:
                continue
                
            delta_x = np.max(pixel_x) - np.min(pixel_x)
            delta_y = np.max(pixel_y) - np.min(pixel_y)
            line_extent = max(delta_x, delta_y)
            
            # --- THE CORRECTED FILTER GATE ---
            if line_extent >= min_line_length:
                boosted_confidence[pixel_y, pixel_x] = 1.0
            else:
                # CRITICAL: This explicitly forces small fragments to disappear
                boosted_confidence[pixel_y, pixel_x] = 0.0
                
        return boosted_confidence

    def process_patch(self, gray_patch, min_line_length=18):
        """Main execution sequence with diagnostics."""
        
        gray_patch = gray_patch[:self.target_size, :self.target_size]
        ridge_map = self._extract_confidence_field(gray_patch)

        candidate_mask = ridge_map > 0.22
        plt.figure(figsize=(8,8))
        plt.imshow(candidate_mask, cmap='gray')
        plt.title("Candidate Mask")
        plt.show()

        print("\n========== STAGE 1 ==========")
        print(f"Candidate pixels: {np.count_nonzero(candidate_mask)}")

        skeleton = skeletonize(candidate_mask)
        skel_rows, skel_cols = np.where(skeleton)
        skeleton_pts = np.column_stack((skel_cols, skel_rows))

        print("\n========== STAGE 2 ==========")
        print(f"Skeleton pixels: {len(skeleton_pts)}")

        if len(skeleton_pts) == 0:
            print("No skeleton points found.")
            return np.zeros_like(gray_patch)

        classified_features, tangent_map, pixel_confidence = self._analyze_topology(
            skeleton_pts,
            ridge_map,
            gray_patch
        )

        print("\n========== STAGE 3 ==========")
        print(f"Accepted structural points: {len(tangent_map)}")
        print(f"Junctions: {sum(v == 'junction' for v in classified_features.values())}")
        print(f"Lines: {sum(v == 'line' for v in classified_features.values())}")

        raw_mask, cleaned_mask, final_skel = self._stitch_and_generate_confidence_map(
            classified_features,
            tangent_map,
            pixel_confidence,
            ridge_map
        )

        print("\n========== STAGE 4 ==========")
        print(f"Pixels after stitching: {np.count_nonzero(final_skel)}")

        # Diagnostics BEFORE sieve
        pre_labels, pre_count = ndimage.label(final_skel)
        print(f"Connected components before sieve: {pre_count}")

        for comp_id in range(1, min(pre_count + 1, 15)):
            ys, xs = np.where(pre_labels == comp_id)
            component_pixels = len(xs)
            delta_x = xs.max() - xs.min()
            delta_y = ys.max() - ys.min()
            extent = max(delta_x, delta_y)

            print(
                f"Component {comp_id:3d} | "
                f"Pixels={component_pixels:5d} | "
                f"Extent={extent:4d}"
            )

        final_confidence_map = self._apply_topological_length_sieve(
            final_skel.astype(float),
            min_line_length=min_line_length
        )

        print("\n========== STAGE 5 ==========")
        post_mask = final_confidence_map > 0.05
        post_labels, post_count = ndimage.label(post_mask)

        print(f"Connected components after sieve: {post_count}")
        print(f"Remaining pixels: {np.count_nonzero(post_mask)}")

        return final_confidence_map


# =========================================================================
# RUNNER WITH TUNABLE LENGTH SIEVE SWITCH (STANDALONE GLOBAL FUNCTION)
# =========================================================================
def execute_tuned_confidence_run_new(geojson_path, plot_no, tiff_imagery="imagery.tif", min_confidence=0.42, min_line_length=18):
    """Runs extraction, labels individual connected components on the graph, and filters by length."""
    target_size = 500
    gdf = gpd.read_file(geojson_path)
    target_row = gdf.iloc[[plot_no]].copy()
    
    with rasterio.open(tiff_imagery) as src:
        if target_row.crs != src.crs:
            target_row = target_row.to_crs(src.crs)
        raw_geom = target_row.geometry.values[0]
        vector_poly = max(raw_geom.geoms, key=lambda p: p.area) if isinstance(raw_geom, MultiPolygon) else raw_geom
        
        centroid = vector_poly.centroid
        hw, hh = (target_size / 2) * abs(src.transform.a), (target_size / 2) * abs(src.transform.e)
        bbox = box(centroid.x - hw, centroid.y - hh, centroid.x + hw, centroid.y + hh)
        cropped_raster, _ = rasterio.mask.mask(src, [bbox], crop=True)
        
    if cropped_raster.shape[0] >= 3:
        img_patch = np.moveaxis(cropped_raster[:3], 0, -1).astype(float)
        gray_patch = (img_patch[:, :, 0] * 0.299 + img_patch[:, :, 1] * 0.587 + img_patch[:, :, 2] * 0.114)
    else:
        gray_patch = cropped_raster[0].astype(float)
        img_patch = np.stack([gray_patch, gray_patch, gray_patch], axis=-1)
        
    img_patch = img_patch[:target_size, :target_size]
    gray_patch = gray_patch[:target_size, :target_size]
    
    if gray_patch.max() > gray_patch.min():
        gray_patch = (gray_patch - gray_patch.min()) / (gray_patch.max() - gray_patch.min())
    if img_patch.max() > 0:
        img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min())

    # 1. Generate Raw Structural Map Up To Stitching Stage
    engine = CadastralConfidenceEngine(target_size=target_size)
    gray_patch_clipped = gray_patch[:engine.target_size, :engine.target_size]
    ridge_map = engine._extract_confidence_field(gray_patch_clipped)
    
    # Tightened threshold gate pass matching stage 1
    candidate_mask = ridge_map > 0.42  
    skeleton = skeletonize(candidate_mask)
    skel_rows, skel_cols = np.where(skeleton)
    skeleton_pts = np.column_stack((skel_cols, skel_rows))
    
    if len(skeleton_pts) == 0:
        print("No structural features extracted.")
        return
        
    classified_features, tangent_map, pixel_confidence = engine._analyze_topology(skeleton_pts, ridge_map, gray_patch_clipped)
    
    # Capture the full multi-mask tuple from our clean sieve-then-stitch design
    raw_mask, cleaned_mask, final_skel = engine._stitch_and_generate_confidence_map(
        classified_features, tangent_map, pixel_confidence, ridge_map
    )
    
    # 2. Compute Connected Components BEFORE the sieve for Graph Labeling
    labeled_mask, num_features = ndimage.label(final_skel, structure=np.ones((3, 3)))
    
    # 3. Map overlay system to output display configurations
    overlay_display = np.copy(img_patch)
    overlay_display[final_skel] = [0.0, 1.0, 0.0] 
    
    # 4. Generate the Plot Windows
    plt.close('all') 
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), clear=True)
    
    # Subplot 1: The noisy target visualization matrix
    axes[0].imshow(raw_mask, cmap='gray')
    axes[0].set_title("1. Raw Candidate Mask\n(With Noise)")
    axes[0].axis('off')
    
    # Subplot 2: Cleaned footprint matrix following pixel sieve passes
    axes[1].imshow(cleaned_mask, cmap='gray')
    axes[1].set_title("2. Cleaned Mask\n(Post Pixel-Sieve)")
    axes[1].axis('off')
    
    # Identify the largest component to avoid filling the graph with identical numbers
    component_sizes = [np.sum(labeled_mask == fid) for fid in range(1, num_features + 1)]
    largest_component_id = np.argmax(component_sizes) + 1 if component_sizes else -1

    # Print Component Locations onto the Figure canvas
    for feature_id in range(1, min(num_features + 1, 150)):  # Caps text rendering at 150 traces for performance
        pixel_y, pixel_x = np.where(labeled_mask == feature_id)
        if len(pixel_x) == 0:
            continue
            
        # Place label text at the geometric center of the component string
        mean_x = int(np.mean(pixel_x))
        mean_y = int(np.mean(pixel_y))
        
        # Color code: Cyan for small components, Blue for the massive interconnected web
        text_color = '#00FFFF' if feature_id != largest_component_id else '#3388FF'
        font_weight = 'bold' if feature_id == largest_component_id else 'normal'
        
        axes[1].text(
            mean_x, mean_y, str(feature_id),
            color=text_color, fontsize=7, weight=font_weight,
            ha='center', va='center',
            bbox=dict(boxstyle='square,pad=0.1', fc='black', ec='none', alpha=0.6)
        )
    
    # Subplot 3: Final Output Output Display
    axes[2].imshow(overlay_display)
    axes[2].set_title(f"3. Protected Boundaries\n(Clean E2E Bridges)")
    axes[2].axis('off')
    
    plt.tight_layout()
    fig.canvas.draw()
    plt.show(block=True)

    # Secondary plot keeping consistency with past verification configurations
    plt.figure(figsize=(8,8))
    plt.imshow(candidate_mask, cmap='gray')
    plt.title("Candidate Mask")
    plt.show()


def run_extraction(
    input_tif="imagery.tif",
    output_tif="extracted.tif"
):

    engine = CadastralConfidenceEngine(target_size=5000)

    with rasterio.open(input_tif) as src:

        img = src.read(1).astype(np.float32)

        if img.max() > img.min():
            img = (img - img.min()) / (img.max() - img.min())

        extracted = engine.process_patch(img)

        profile = src.profile.copy()

        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress="lzw"
        )

        with rasterio.open(output_tif, "w", **profile) as dst:

            dst.write(
                (extracted > 0.05).astype(np.uint8),
                1
            )

    print(f"Saved {output_tif}")

# =========================================================================
# RUN SCRIPT ENTRY POINT
# =========================================================================
if __name__ == "__main__":
    print("Extraction starts here...")
    run_extraction(
    "imagery.tif",
    "extracted.tif"
)