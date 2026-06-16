import geopandas as gpd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt

jsondata = gpd.read_file("input.geojson")
min_lat, max_lat, min_lon, max_lon = (jsondata.geometry[0].bounds)
print(min_lat, max_lat, min_lon, max_lon)
min_lat, max_lat, min_lon, max_lon = (jsondata.geometry[1].bounds)
print(min_lat, max_lat, min_lon, max_lon)
min_lat, max_lat, min_lon, max_lon = (jsondata.geometry[2].bounds)
print(min_lat, max_lat, min_lon, max_lon)




import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import matplotlib.pyplot as plt

# --- 1. Fix the Unpacking Order Here ---
# Shapely bounds always output: min_x (lon), min_y (lat), max_x (lon), max_y (lat)
min_lon, min_lat, max_lon, max_lat = jsondata.geometry[0].bounds


def plot_tiff_by_coordinates(min_lat, max_lat, min_lon, max_lon, tiff_path="boundaries.tif"):
    """
    Crops a TIFF image using GPS coordinates (lat/lon bounds) and plots the result.
    """
    # Create the bounding box using correct axis assignments
    # box format: box(min_x, min_y, max_x, max_y) -> (min_lon, min_lat, max_lon, max_lat)
    geo_box = box(min_lon, min_lat, max_lon, max_lat)
    
    gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[geo_box])
    
    with rasterio.open(tiff_path) as src:
        gdf_projected = gdf.to_crs(src.crs)
        geometries = gdf_projected.geometry.values
        
        try:
            # Crop the image to the bounding box coordinates
            cropped_image, cropped_transform = mask(src, geometries, crop=True)
            band_to_plot = cropped_image[0]
            
            plt.figure(figsize=(8, 8))
            
            # --- 2. Fix the Extent Mapping Matrix Here ---
            # total_bounds outputs: xmin, ymin, xmax, ymax
            xmin, ymin, xmax, ymax = gdf_projected.total_bounds
            
            # imshow extent must strictly follow: [xmin, xmax, ymin, ymax]
            plt.imshow(band_to_plot, cmap='viridis', extent=[xmin, xmax, ymin, ymax])
            
            plt.title(f"Cropped Region\nLat: [{min_lat:.5f}, {max_lat:.5f}] | Lon: [{min_lon:.5f}, {max_lon:.5f}]")
            plt.xlabel("X Coordinate (Meters East)")
            plt.ylabel("Y Coordinate (Meters North)")
            plt.colorbar(label="Pixel Value (0-255)")
            # plt.show()
            
        except ValueError:
            print("Error: The provided coordinates do not overlap with the TIFF file's geographic boundary.")

# --- 3. Run the Function with Correct Order ---
jsondata = gpd.read_file("input.geojson")



# Loop through your geometries cleanly using the right bound assignment

# print(f"\n--- Plotting Geometry {0} ---")
# b_lon_min, b_lat_min, b_lon_max, b_lat_max = jsondata.geometry[0].bounds

# plot_tiff_by_coordinates(
#     min_lat=b_lat_min, 
#     max_lat=b_lat_max, 
#     min_lon=b_lon_min, 
#     max_lon=b_lon_max
# )


import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from shapely.affinity import translate
from shapely.geometry import box

import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.plot import show as raster_show
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.plot import show as raster_show
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.plot import show as raster_show
import matplotlib.pyplot as plt

def plot_separated_reference_zoomed(geojson_path, plot_no, tiff_path="boundaries.tif", zoom_out_factor=0.3):
    """
    Crops a TIFF using the GeoJSON feature geometry, zooms out slightly by adding a buffer margin,
    and marks the matching centroids on two separate, synchronized plots.
    
    Parameters:
    zoom_out_factor (float): Percentage to zoom out (e.g., 0.3 adds 30% padding around the bounding box).
    """
    # 1. Read the GeoJSON map
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]]
    
    # 2. Open the imagery raster file
    src_imagery = rasterio.open(tiff_path)
    
    # 3. Align Coordinate Systems
    if target_plot.crs != src_imagery.crs:
        target_plot = target_plot.to_crs(src_imagery.crs)
        
    # 4. Crop the image down to just the plot's bounding box
    shapes = target_plot.geometry.values
    cropped_image, cropped_transform = rasterio.mask.mask(src_imagery, shapes, crop=True)
    
    # 5. Calculate the core spatial window limits from the cropped transform matrix
    xmin = cropped_transform.c
    ymax = cropped_transform.f
    xmax = xmin + (cropped_image.shape[2] * cropped_transform.a)
    ymin = ymax + (cropped_image.shape[1] * cropped_transform.e)
    
    # 6. Calculate a padding margin to zoom out slightly
    width_padding = (xmax - xmin) * zoom_out_factor
    height_padding = (ymax - ymin) * zoom_out_factor
    
    xmin_zoomed = xmin - width_padding
    xmax_zoomed = xmax + width_padding
    ymin_zoomed = ymin - height_padding
    ymax_zoomed = ymax + height_padding

    # 7. Calculate the true geometric centroid of the plot feature
    centroid = target_plot.geometry.centroid.values[0]
    centroid_x, centroid_y = centroid.x, centroid.y

    # 8. Setup the side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Graph 1: The Isolated Raster Imagery Cutout ---
    raster_show(cropped_image, ax=ax1, transform=cropped_transform, cmap='gray')
    # Mark the centroid on the image
    ax1.plot(centroid_x, centroid_y, marker='X', color='red', markersize=12, label='Centroid Target')
    
    # Apply the zoomed-out window limits to the image panel
    ax1.set_xlim(xmin_zoomed, xmax_zoomed)
    ax1.set_ylim(ymin_zoomed, ymax_zoomed)
    
    ax1.set_title(f"Imagery Focus (Index {plot_no})")
    ax1.set_xlabel("Meters East")
    ax1.set_ylabel("Meters North")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # --- Graph 2: The Vector Boundary Shape ---
    target_plot.plot(ax=ax2, facecolor='none', edgecolor='cyan', linewidth=2)
    # Mark the matching centroid on the vector plot
    ax2.plot(centroid_x, centroid_y, marker='X', color='red', markersize=12, label='Centroid Target')
    
    # Apply identical zoomed-out window limits to the vector panel
    ax2.set_xlim(xmin_zoomed, xmax_zoomed)
    ax2.set_ylim(ymin_zoomed, ymax_zoomed)
    
    ax2.set_title(f"Vector Reference (Index {plot_no})")
    ax2.set_xlabel("Meters East")
    ax2.set_ylabel("Meters North")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    
    src_imagery.close()
    return fig

# plot_separated_reference_zoomed("input.geojson",300)

import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.plot import show as raster_show
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.plot import show as raster_show
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.plot import show as raster_show
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.plot import show as raster_show
import matplotlib.pyplot as plt

import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.plot import show as raster_show
import matplotlib.pyplot as plt
import numpy as np

def plot_three_way_grayscale_enhanced(geojson_path, plot_no, tiff_path_1="boundaries.tif", tiff_path_2="imagery.tif", image_buffer_meters=100, zoom_out_factor=0.5):
    """
    Crops two separate TIFF files. Converts a multi-band color image (img.tif) 
    into a true 2D grayscale array using a luminosity formula to enhance line extraction.
    """
    # 1. Read the GeoJSON map
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]]
    
    # 2. Open both imagery raster files
    src_boundaries = rasterio.open(tiff_path_1)
    src_img = rasterio.open(tiff_path_2)
    
    # 3. Align Coordinate Systems
    if target_plot.crs != src_boundaries.crs:
        target_plot = target_plot.to_crs(src_boundaries.crs)
        
    # 4. Expand the masking area using a buffer (in meters)
    buffered_shape = target_plot.geometry.buffer(image_buffer_meters).values
    
    # Crop both images using the exact same shape context
    cropped_boundaries, transform_boundaries = rasterio.mask.mask(src_boundaries, buffered_shape, crop=True)
    cropped_img, transform_img = rasterio.mask.mask(src_img, buffered_shape, crop=True)
    
    # 5. TRUE GRAYSCALE CONVERSION FOR THE IMAGERY
    # If the image has 3 channels (RGB), calculate weighted luminosity: Y = 0.299R + 0.587G + 0.114B
    if cropped_img.shape[0] >= 3:
        raw_gray = (0.299 * cropped_img[0] + 
                    0.587 * cropped_img[1] + 
                    0.114 * cropped_img[2])
        
        # Contrast Enhancement: Normalize the pixel values to use the full 0-255 dynamic range
        p_min, p_max = np.percentile(raw_gray, (2, 98)) # Clip outliers to avoid glare/deep shadows
        enhanced_gray = np.clip((raw_gray - p_min) / (p_max - p_min) * 255, 0, 255)
    else:
        # If it's already a single-channel band, just take it directly
        enhanced_gray = cropped_img[0]
    
    # 6. Calculate window limits from the crop transform matrix
    xmin = transform_boundaries.c
    ymax = transform_boundaries.f
    xmax = xmin + (cropped_boundaries.shape[2] * transform_boundaries.a)
    ymin = ymax + (cropped_boundaries.shape[1] * transform_boundaries.e)
    
    # 7. Apply identical zoom padding limits
    width_padding = (xmax - xmin) * zoom_out_factor
    height_padding = (ymax - ymin) * zoom_out_factor
    
    xmin_zoomed = xmin - width_padding
    xmax_zoomed = xmax + width_padding
    ymin_zoomed = ymin - height_padding
    ymax_zoomed = ymax + height_padding

    # 8. Calculate the target centroid coordinate
    centroid = target_plot.geometry.centroid.values[0]
    centroid_x, centroid_y = centroid.x, centroid.y

    # 9. Setup the 3-panel side-by-side layout canvas
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Graph 1: Boundaries.tif Output ---
    raster_show(cropped_boundaries, ax=ax1, transform=transform_boundaries, cmap='gray')
    ax1.plot(centroid_x, centroid_y, marker='X', color='red', markersize=12, label='Centroid Target')
    ax1.set_xlim(xmin_zoomed, xmax_zoomed)
    ax1.set_ylim(ymin_zoomed, ymax_zoomed)
    ax1.set_title("Boundaries Reference (Grayscale)")
    ax1.set_xlabel("Meters East")
    ax1.set_ylabel("Meters North")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # --- Graph 2: Img.tif Output (True Grayscale + High Contrast Enhanced) ---
    # We pass a pure 2D array directly to ax.imshow so cmap='gray' forces a crisp black & white image
    ax2.imshow(enhanced_gray, cmap='gray', extent=[xmin, xmax, ymin, ymax])
    ax2.plot(centroid_x, centroid_y, marker='X', color='red', markersize=12, label='Centroid Target')
    ax2.set_xlim(xmin_zoomed, xmax_zoomed)
    ax2.set_ylim(ymin_zoomed, ymax_zoomed)
    ax2.set_title("Enhanced Grayscale Base Imagery")
    ax2.set_xlabel("Meters East")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # --- Graph 3: Vector Line Layout ---
    gdf.to_crs(src_boundaries.crs).plot(ax=ax3, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=1)
    target_plot.plot(ax=ax3, facecolor='none', edgecolor='cyan', linewidth=2.5, label='Target Plot')
    ax3.plot(centroid_x, centroid_y, marker='X', color='red', markersize=12, label='Centroid Target')
    ax3.set_xlim(xmin_zoomed, xmax_zoomed)
    ax3.set_ylim(ymin_zoomed, ymax_zoomed)
    ax3.set_title("Vector Boundary Framework")
    ax3.set_xlabel("Meters East")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    
    # Close streams cleanly
    src_boundaries.close()
    src_img.close()
    
    return fig

# plot_three_way_grayscale_enhanced("truth.geojson",1)

import rasterio
import rasterio.mask
import numpy as np

def extract_bright_features_inverted(tiff_path, geometry_shape, image_buffer_meters=100, dark_threshold_percentile=25):
    """
    Crops a TIFF image, applies local adaptive histogram equalization (CLAHE), 
    and inverts it so lines are bright white on a clean black background canvas.
    """
    # 1. Open and crop the image
    with rasterio.open(tiff_path) as src:
        buffered_shape = geometry_shape.buffer(image_buffer_meters)
        cropped_image, cropped_transform = rasterio.mask.mask(src, [buffered_shape], crop=True)
        
    # 2. Convert multi-band RGB to a single 2D Grayscale array
    if cropped_image.shape[0] >= 3:
        gray_image = (0.299 * cropped_image[0] + 
                      0.587 * cropped_image[1] + 
                      0.114 * cropped_image[2])
    else:
        gray_image = cropped_image[0].astype(float)
        
    # 3. Enhance Contrast
    p_min, p_max = np.percentile(gray_image, (1, 99))
    if p_max - p_min == 0: p_max += 1
    normalized_img = np.clip((gray_image - p_min) / (p_max - p_min), 0.0, 1.0)
    
    enhanced_gray = exposure.equalize_adapthist(normalized_img, clip_limit=0.04) * 255
    
    # 4. Invert values: Dark boundaries become maximum brightness values
    inverted_gray = 255.0 - enhanced_gray
    
    # 5. Thresholding the inverted image
    # Because the image is inverted, our target features are now the BRIGHTEST values.
    # We turn everything below the top percentile threshold to 0 (Pure Black background).
    cutoff_value = np.percentile(inverted_gray, (100 - dark_threshold_percentile))
    
    filtered_inverted = inverted_gray.copy()
    filtered_inverted[filtered_inverted < cutoff_value] = 0
    
    return filtered_inverted, cropped_transform


import geopandas as gpd
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from rasterio.plot import show as raster_show
from skimage import exposure

def plot_three_way_grayscale_inverted(geojson_path, plot_no, tiff_path_1="boundaries.tif", tiff_path_2="imagery.tif", image_buffer_meters=100, zoom_out_factor=0.5):
    """
    Crops two separate TIFF files. Converts a multi-band color image (img.tif) 
    into an ultra-high contrast, INVERTED 2D grayscale array (White lines, Black background).
    """
    # 1. Read the GeoJSON map
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]]
    
    # 2. Open both imagery raster files safely
    with rasterio.open(tiff_path_1) as src_boundaries, rasterio.open(tiff_path_2) as src_img:
        
        # 3. Align Coordinate Systems
        if target_plot.crs != src_boundaries.crs:
            target_plot = target_plot.to_crs(src_boundaries.crs)
            
        # 4. Expand the masking area using a buffer (in meters)
        buffered_shape = target_plot.geometry.buffer(image_buffer_meters).values
        
        # Crop both images using the exact same shape context
        cropped_boundaries, transform_boundaries = rasterio.mask.mask(src_boundaries, buffered_shape, crop=True)
        cropped_img, transform_img = rasterio.mask.mask(src_img, buffered_shape, crop=True)
    
    # 5. TRUE GRAYSCALE CONVERSION
    if cropped_img.shape[0] >= 3:
        raw_gray = (0.299 * cropped_img[0] + 
                    0.587 * cropped_img[1] + 
                    0.114 * cropped_img[2])
    else:
        raw_gray = cropped_img[0].astype(float)
        
    # --- MAX ADAPTIVE CONTRAST AND INVERSION ENGINE ---
    # Step A: Tight min-max normalization to strip out extreme specular glare
    g_min, g_max = np.percentile(raw_gray, (1, 99))
    if g_max - g_min == 0: g_max += 1
    normalized_base = np.clip((raw_gray - g_min) / (g_max - g_min), 0.0, 1.0)
    
    # Step B: Run local CLAHE enhancement
    enhanced_gray = exposure.equalize_adapthist(normalized_base, clip_limit=0.04) * 255
    
    # Step C: INVERT THE COLORS (Makes lines White [255], background Black [0])
    inverted_gray = 255.0 - enhanced_gray
    
    # 6. Calculate window limits from the crop transform matrix
    xmin = transform_boundaries.c
    ymax = transform_boundaries.f
    xmax = xmin + (cropped_boundaries.shape[2] * transform_boundaries.a)
    ymin = ymax + (cropped_boundaries.shape[1] * transform_boundaries.e)
    
    # 7. Apply identical zoom padding limits
    width_padding = (xmax - xmin) * zoom_out_factor
    height_padding = (ymax - ymin) * zoom_out_factor
    
    xmin_zoomed = xmin - width_padding
    xmax_zoomed = xmax + width_padding
    ymin_zoomed = ymin - height_padding
    ymax_zoomed = ymax + height_padding

    # 8. Calculate the target centroid coordinate
    centroid = target_plot.geometry.centroid.values[0]
    centroid_x, centroid_y = centroid.x, centroid.y

    # 9. Setup the 3-panel side-by-side layout canvas
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # --- Graph 1: Boundaries.tif Output ---
    raster_show(cropped_boundaries, ax=ax1, transform=transform_boundaries, cmap='gray')
    ax1.plot(centroid_x, centroid_y, marker='X', color='red', markersize=12, label='Centroid Target')
    ax1.set_xlim(xmin_zoomed, xmax_zoomed)
    ax1.set_ylim(ymin_zoomed, ymax_zoomed)
    ax1.set_title("Boundaries Reference (Grayscale)")
    ax1.set_xlabel("Meters East")
    ax1.set_ylabel("Meters North")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # --- Graph 2: Img.tif Output (Inverted: White Lines on Black Space) ---
    ax2.imshow(inverted_gray, cmap='gray', extent=[xmin, xmax, ymin, ymax])
    ax2.plot(centroid_x, centroid_y, marker='X', color='red', markersize=12, label='Centroid Target')
    ax2.set_xlim(xmin_zoomed, xmax_zoomed)
    ax2.set_ylim(ymin_zoomed, ymax_zoomed)
    ax2.set_title("Inverted High-Contrast Imagery")
    ax2.set_xlabel("Meters East")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    # --- Graph 3: Vector Line Layout ---
    gdf.to_crs(src_boundaries.crs).plot(ax=ax3, facecolor='none', edgecolor='gray', alpha=0.5, linewidth=1)
    target_plot.plot(ax=ax3, facecolor='none', edgecolor='cyan', linewidth=2.5, label='Target Plot')
    ax3.plot(centroid_x, centroid_y, marker='X', color='red', markersize=12, label='Centroid Target')
    ax3.set_xlim(xmin_zoomed, xmax_zoomed)
    ax3.set_ylim(ymin_zoomed, ymax_zoomed)
    ax3.set_title("Vector Boundary Framework")
    ax3.set_xlabel("Meters East")
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.show()
    return fig



import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, shape
from shapely.affinity import translate
from skimage import measure

import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate
from skimage import measure

def calculate_centroid_aligned_iou(geojson_path, plot_no, tiff_path_1="boundaries.tif", image_buffer_meters=50):
    """
    Aligns the centroids of the GeoJSON polygon and the raster feature exactly,
    computes the Intersection over Union (IoU) distortion metric, and plots the overlap.
    Safely handles both Polygon and MultiPolygon vector types.
    """
    # 1. Load Vector Data
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]].copy()
    raw_vector_geom = target_plot.geometry.values[0]
    
    # --- SAFE GEOMETRY EXTRACTOR ---
    # If it's a MultiPolygon, extract only the largest constituent polygon by area
    if isinstance(raw_vector_geom, MultiPolygon):
        print(f"[NOTE] Plot index {plot_no} is a MultiPolygon. Extracting the largest continuous piece...")
        vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area)
    else:
        vector_poly = raw_vector_geom

    vector_centroid = vector_poly.centroid
    
    # 2. Extract and Clean the Raster Feature
    with rasterio.open(tiff_path_1) as src:
        if target_plot.crs != src.crs:
            target_plot = target_plot.to_crs(src.crs)
            # Re-extract and re-check geometry types after CRS transformation
            raw_vector_geom = target_plot.geometry.values[0]
            if isinstance(raw_vector_geom, MultiPolygon):
                vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area)
            else:
                vector_poly = raw_vector_geom
            vector_centroid = vector_poly.centroid
            
        buffered_shape = vector_poly.buffer(image_buffer_meters)
        cropped_raster, cropped_transform = rasterio.mask.mask(src, [buffered_shape], crop=True)
    
    # Convert to 2D binary array (assuming dark lines/features are < 128)
    if cropped_raster.shape[0] >= 3:
        gray = (0.299 * cropped_raster[0] + 0.587 * cropped_raster[1] + 0.114 * cropped_raster[2])
    else:
        gray = cropped_raster[0].astype(float)
        
    # Threshold to create a clean binary mask of the target plot feature
    binary_raster = (gray < 128).astype(np.uint8)
    
    # 3. Extract the Vector Geometry of the Raster Feature to find its true Centroid
    contours = measure.find_contours(binary_raster, 0.5)
    if len(contours) == 0:
        print(f"[ERROR] No raster boundaries detected in TIFF for plot index {plot_no}")
        return None
        
    # Take the largest continuous boundary contour found in the crop box
    largest_contour = max(contours, key=len)
    
    # Convert pixel contour coordinates back into map coordinates (meters)
    raster_coords_meters = []
    for row, col in largest_contour:
        x, y = cropped_transform * (col, row)
        raster_coords_meters.append((x, y))
        
    raster_poly = Polygon(raster_coords_meters)
    raster_centroid = raster_poly.centroid
    
    # 4. LOCK CENTROIDS: Shift the Vector Polygon exactly onto the Raster Centroid
    dX = raster_centroid.x - vector_centroid.x
    dY = raster_centroid.y - vector_centroid.y
    aligned_vector_poly = translate(vector_poly, xoff=dX, yoff=dY)
    
    # 5. Compute the Intersection over Union (IoU) Metrics
    intersection_area = aligned_vector_poly.intersection(raster_poly).area
    union_area = aligned_vector_poly.union(raster_poly).area
    iou_score = intersection_area / union_area if union_area > 0 else 0
    
    print(f"\n--- Diagnostic Metrics for Plot Index {plot_no} ---")
    print(f"Initial Positional Displacement Offset: dX = {dX:.2f}m, dY = {dY:.2f}m")
    print(f"Shape Distorted Overlap Score (Centroid Aligned IoU): {iou_score:.4f}")
    
    # 6. Plot the Alignment Diagnosis Frame
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot Raster Polygon base reference
    x_rast, y_rast = raster_poly.exterior.xy
    ax.fill(x_rast, y_rast, alpha=0.3, fc='gold', ec='darkgoldenrod', linewidth=2, label='True Raster Shape')
    
    # Plot Centroid-Aligned Vector Polygon (Now safely calling .exterior)
    x_vec, y_vec = aligned_vector_poly.exterior.xy
    ax.plot(x_vec, y_vec, color='cyan', linewidth=2.5, linestyle='-', label='Centroid-Aligned Vector')
    
    # Plot the Shared Static Anchor Center point
    ax.plot(raster_centroid.x, raster_centroid.y, marker='o', color='red', markersize=10, label='Locked Centroid Anchor')
    
    # Add Metric Box overlay on canvas
    metric_text = f"Centroid Aligned IoU: {iou_score:.2%}\nOffset Vector: [{dX:.1f}m, {dY:.1f}m]"
    box_color = "lightgreen" if iou_score >= 0.80 else "orange" if iou_score >= 0.65 else "coral"
    ax.text(0.03, 0.97, metric_text, transform=ax.transAxes, fontsize=11, fontweight='bold',
            family='monospace', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.85))
    
    ax.set_title(f"Shape Distortion Evaluation (Plot {plot_no})", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.show()
    return iou_score

# iou = calculate_centroid_aligned_iou("input.geojson",2 )
# # plot_three_way_grayscale_enhanced("input.geojson",2)
# print(iou)



import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate
from skimage import measure

def plot_comprehensive_area_diagnosis(geojson_path, plot_no, tiff_path_1="boundaries.tif", recorded_area_col="recorded_area_sqm", image_buffer_meters=60):
    """
    Aligns centroids, extracts the true raster feature area, and overlays
    the raw vector area, registry area, and raster area directly onto the plot canvas.
    """
    # 1. Load Vector Data and Extract Areas
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]].copy()
    raw_vector_geom = target_plot.geometry.values[0]
    
    # Extract official recorded area from the GeoJSON attribute row
    if recorded_area_col in target_plot.columns:
        registry_area = float(target_plot[recorded_area_col].values[0])
    else:
        registry_area = 0.0  # Fallback if column name doesn't match
    
    # Safe Polygon extraction for area calculations
    if isinstance(raw_vector_geom, MultiPolygon):
        vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area)
    else:
        vector_poly = raw_vector_geom

    vector_centroid = vector_poly.centroid
    vector_calculated_area = vector_poly.area  # Geometry area before alignment
    
    # 2. Extract and Process the Raster Feature
    with rasterio.open(tiff_path_1) as src:
        if target_plot.crs != src.crs:
            target_plot = target_plot.to_crs(src.crs)
            raw_vector_geom = target_plot.geometry.values[0]
            vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area) if isinstance(raw_vector_geom, MultiPolygon) else raw_vector_geom
            vector_centroid = vector_poly.centroid
            vector_calculated_area = vector_poly.area
            
        buffered_shape = vector_poly.buffer(image_buffer_meters)
        cropped_raster, cropped_transform = rasterio.mask.mask(src, [buffered_shape], crop=True)
    
    # Convert to 2D binary mask
    if cropped_raster.shape[0] >= 3:
        gray = (0.299 * cropped_raster[0] + 0.587 * cropped_raster[1] + 0.114 * cropped_raster[2])
    else:
        gray = cropped_raster[0].astype(float)
        
    binary_raster = (gray < 128).astype(np.uint8)
    
    # 3. Trace the Raster Boundary to calculate its True Physical Ground Area
    contours = measure.find_contours(binary_raster, 0.5)
    if len(contours) == 0:
        print(f"[ERROR] No raster boundaries detected in TIFF for plot index {plot_no}")
        return None
        
    largest_contour = max(contours, key=len)
    raster_coords_meters = [cropped_transform * (col, row) for row, col in largest_contour]
        
    raster_poly = Polygon(raster_coords_meters)
    raster_centroid = raster_poly.centroid
    raster_feature_area = raster_poly.area  # The true ground area from boundaries.tif
    
    # 4. Shift Vector Polygon exactly onto the Raster Centroid for shape comparison
    dX = raster_centroid.x - vector_centroid.x
    dY = raster_centroid.y - vector_centroid.y
    aligned_vector_poly = translate(vector_poly, xoff=dX, yoff=dY)
    
    # Calculate Centroid-Aligned IoU
    intersection_area = aligned_vector_poly.intersection(raster_poly).area
    union_area = aligned_vector_poly.union(raster_poly).area
    iou_score = intersection_area / union_area if union_area > 0 else 0
    
    # 5. Build the Diagnostic Plot Canvas
    fig, ax = plt.subplots(figsize=(9, 10))
    
    # Plot true physical raster layout base
    x_rast, y_rast = raster_poly.exterior.xy
    ax.fill(x_rast, y_rast, alpha=0.25, fc='gold', ec='darkgoldenrod', linewidth=2, label='True Raster Shape')
    
    # Plot centroid-aligned vector polygon overlay
    x_vec, y_vec = aligned_vector_poly.exterior.xy
    ax.plot(x_vec, y_vec, color='cyan', linewidth=2.5, label='Centroid-Aligned Vector')
    
    # Mark the shared centroid point
    ax.plot(raster_centroid.x, raster_centroid.y, marker='o', color='red', markersize=10, label='Locked Centroid Anchor')
    
    # --- DYNAMIC CANVAS TEXT BOX ---
    # Compiling all requested spatial area metrics side-by-side
    metric_text = (
        f"DIAGNOSTIC METRICS (PLOT {plot_no})\n"
        f"---------------------------------\n"
        f"1. Registry Record Area : {registry_area:,.2f} sqm\n"
        f"2. GeoJSON Vector Area  : {vector_calculated_area:,.2f} sqm\n"
        f"3. True Raster Line Area: {raster_feature_area:,.2f} sqm\n"
        f"---------------------------------\n"
        f"Centroid Aligned IoU    : {iou_score:.2%}\n"
        f"Offset Translation Vector: [{dX:+.1f}m, {dY:+.1f}m]"
    )
    
    # Color-code box based on safety threshold
    box_color = "lightgreen" if iou_score >= 0.80 else "orange" if iou_score >= 0.65 else "coral"
    
    ax.text(0.03, 0.97, metric_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
            family='monospace', va='top', bbox=dict(boxstyle='round,pad=0.6', facecolor=box_color, alpha=0.9))
    
    ax.set_title(f"Area Evaluation Framework — Plot {plot_no}", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return registry_area, vector_calculated_area, raster_feature_area

# Run the complete metric dashboard pass
# reg_a, vec_a, rast_a = plot_comprehensive_area_diagnosis("input.geojson", 300, recorded_area_col="recorded_area_sqm")


# jsondata = gpd.read_file("input.geojson")
print(jsondata.head(0))

# from bhume import score
# score("predictions.geojson",)


import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.affinity import translate
from skimage import feature, exposure, measure

def analyze_plot_using_only_imagery(geojson_path, plot_no, tiff_path="imagery.tif", recorded_area_col="recorded_area_sqm", image_buffer_meters=60):
    """
    Exclusively uses imagery.tif to isolate field boundaries via edge gradients.
    Aligns vector-to-imagery centroids, calculates shape IoU, and outputs the structured audit log.
    """
    # 1. Initialize Tracking Payload Structure
    payload = {
        "plot_number": int(plot_no),
        "status": "flagged",
        "confidence": 0.0,
        "method_note": "",
        "geometry": None
    }
    
    # 2. Load Vector Layer and Target Metadata
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]].copy()
    raw_vector_geom = target_plot.geometry.values[0]
    
    if recorded_area_col in target_plot.columns:
        registry_area = float(target_plot[recorded_area_col].values[0])
    else:
        registry_area = raw_vector_geom.area
        
    # Handle MultiPolygon splits cleanly by pulling the primary parcel footprint
    if isinstance(raw_vector_geom, MultiPolygon):
        vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area)
    else:
        vector_poly = raw_vector_geom
    
    vector_centroid = vector_poly.centroid
    payload["geometry"] = mapping(raw_vector_geom) # Default state backup
    
    # 3. Stream Local Bounding Box from Imagery
    with rasterio.open(tiff_path) as src:
        if target_plot.crs != src.crs:
            target_plot = target_plot.to_crs(src.crs)
            raw_vector_geom = target_plot.geometry.values[0]
            vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area) if isinstance(raw_vector_geom, MultiPolygon) else raw_vector_geom
            vector_centroid = vector_poly.centroid
            
        buffered_shape = vector_poly.buffer(image_buffer_meters)
        cropped_raster, cropped_transform = rasterio.mask.mask(src, [buffered_shape], crop=True)
        
    # 4. True Luminosity Grayscale Conversion
    if cropped_raster.shape[0] >= 3:
        gray = (0.299 * cropped_raster[0] + 0.587 * cropped_raster[1] + 0.114 * cropped_raster[2])
    else:
        gray = cropped_raster[0].astype(float)
        
    # 5. Extract Fine Boundary Hints via Local Intensity Gradients
    p_min, p_max = np.percentile(gray, (2, 98))
    if p_max - p_min == 0: p_max += 1
    normalized_gray = np.clip((gray - p_min) / (p_max - p_min), 0.0, 1.0)
    
    # Canny isolation extracts the physical linear ridges of field banks/hedgerows
    edge_hints = feature.canny(normalized_gray, sigma=1.8)
    
    # 6. Reconstruct Closed Feature Boundaries from Isolated Edge Paths
    contours = measure.find_contours(edge_hints.astype(float), 0.5)
    
    if len(contours) == 0:
        payload["method_note"] = "Flagged: Raw imagery contains no discernible edge gradients in this search buffer."
        return payload

    # Extract the prominent structural feature bounding the targeted space
    largest_contour = max(contours, key=len)
    raster_coords_meters = [cropped_transform * (col, row) for row, col in largest_contour]
    
    # Ensure our extracted feature forms a valid geometric loop
    if len(raster_coords_meters) < 3:
        payload["method_note"] = "Flagged: Detected imagery boundaries are fragmental and cannot form a valid polygon footprint."
        return payload
        
    imagery_poly = Polygon(raster_coords_meters)
    imagery_centroid = imagery_poly.centroid
    
    # 7. Execute Precise Linear Coordinate Shifts
    dX = imagery_centroid.x - vector_centroid.x
    dY = imagery_centroid.y - vector_centroid.y
    aligned_vector_poly = translate(vector_poly, xoff=dX, yoff=dY)
    
    # 8. Compute Shape Distortion Metrics (IoU) directly against Imagery Footprint
    intersection_area = aligned_vector_poly.intersection(imagery_poly).area
    union_area = aligned_vector_poly.union(imagery_poly).area
    iou_score = intersection_area / union_area if union_area > 0 else 0
    
    # 9. Evaluate Alignment Integrity Rules
    area_delta_pct = abs(aligned_vector_poly.area - registry_area) / registry_area
    
    if iou_score >= 0.70 and area_delta_pct <= 0.12:
        payload["status"] = "corrected"
        payload["confidence"] = round(float(iou_score), 2)
        payload["method_note"] = f"Aligned purely via imagery edge tracking. Shift vector: dX={dX:.1f}m, dY={dY:.1f}m."
        payload["geometry"] = mapping(aligned_vector_poly)
    else:
        payload["status"] = "flagged"
        payload["confidence"] = round(float(iou_score), 2)
        if area_delta_pct > 0.12:
            payload["method_note"] = f"Flagged: High structural risk. Imagery edge layout area deviates from registry by {area_delta_pct:.1%}."
        else:
            payload["method_note"] = f"Flagged: Weak correlation to imagery lines (IoU: {iou_score:.1%}) after centroid snap."

    # 10. Generate Quality Assurance Visualization Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    xmin, ymax = cropped_transform.c, cropped_transform.f
    xmax = xmin + (cropped_raster.shape[2] * cropped_transform.a)
    ymin = ymax + (cropped_raster.shape[1] * cropped_transform.e)
    
    # Display the genuine background imagery landscape
    ax.imshow(normalized_gray, cmap='gray', extent=[xmin, xmax, ymin, ymax], alpha=0.7)
    
    # Burn the edge detection lines on top as a high-contrast crimson overlay
    edge_mask = np.ma.masked_where(edge_hints == 0, edge_hints)
    ax.imshow(edge_mask, cmap='prism', extent=[xmin, xmax, ymin, ymax], alpha=0.6)
    
    # Plot the vector layout alignment
    x_vec, y_vec = aligned_vector_poly.exterior.xy
    ax.plot(x_vec, y_vec, color='cyan', linewidth=3, label='Centroid-Aligned Vector')
    ax.plot(imagery_centroid.x, imagery_centroid.y, marker='o', color='red', markersize=10, label='Imagery Centroid Anchor')
    
    # Format and present the payload audit box directly onto the grid
    box_text = (
        f"IMAGERY AUDIT ENGINE LOG\n============\n"
        f"Plot ID    : {payload['plot_number']}\n"
        f"Status     : {payload['status'].upper()}\n"
        f"Confidence : {payload['confidence']}\n"
        f"Note       : {payload['method_note']}"
    )
    b_color = "lightgreen" if payload["status"] == "corrected" else "coral"
    ax.text(0.03, 0.97, box_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
            family='monospace', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor=b_color, alpha=0.9))
    
    ax.set_title(f"Pure Imagery Alignment Scan — Plot {plot_no}", fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    plt.show()
    
    return payload


# analyze_plot_using_only_imagery("input.geojson",100)




import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.affinity import translate
from scipy.spatial import distance

def optimize_edges_by_area_target(geojson_path, plot_no, tiff_path="imagery.tif", recorded_area_col="recorded_area_sqm", image_buffer_meters=60, step_size_meters=0.5, max_iterations=20):
    """
    Adjusts individual polygon vertices toward the closest high-contrast black image lines.
    Uses the official recorded registry area as a guiding compass to accept or reject steps.
    """
    # 1. Initialize Tracking Payload
    payload = {
        "plot_number": int(plot_no),
        "status": "flagged",
        "confidence": 0.0,
        "method_note": "",
        "geometry": None
    }
    
    # 2. Load Vector and Target Area Context
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]].copy()
    raw_vector_geom = target_plot.geometry.values[0]
    
    if recorded_area_col in target_plot.columns:
        target_area = float(target_plot[recorded_area_col].values[0])
    else:
        print(f"[ERROR] Registry column '{recorded_area_col}' not found.")
        return payload
        
    if isinstance(raw_vector_geom, MultiPolygon):
        vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area)
    else:
        vector_poly = raw_vector_geom
        
    # 3. Generate Stark Black-and-White Contrast Grid from Imagery
    with rasterio.open(tiff_path) as src:
        if target_plot.crs != src.crs:
            target_plot = target_plot.to_crs(src.crs)
            raw_vector_geom = target_plot.geometry.values[0]
            vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area) if isinstance(raw_vector_geom, MultiPolygon) else raw_vector_geom
            
        buffered_shape = vector_poly.buffer(image_buffer_meters)
        cropped_raster, cropped_transform = rasterio.mask.mask(src, [buffered_shape], crop=True)
        
    if cropped_raster.shape[0] >= 3:
        gray = (0.299 * cropped_raster[0] + 0.587 * cropped_raster[1] + 0.114 * cropped_raster[2])
    else:
        gray = cropped_raster[0].astype(float)
        
    # Push greys to pure black (0) and fields to white (255)
    low_thresh = np.percentile(gray, 15)
    binary_contrast = np.ones_like(gray) * 255
    binary_contrast[gray <= low_thresh] = 0
    
    # Find real-world coordinates of all black line pixels
    black_rows, black_cols = np.where(binary_contrast == 0)
    if len(black_rows) == 0:
        payload["method_note"] = "Flagged: No clear contrast lines found in search window."
        return payload
        
    # Transform black pixel indices into real-world (X, Y) ground coordinates
    black_lines_xyz = [cropped_transform * (c, r) for r, c in zip(black_rows, black_cols)]
    black_lines_xyz = np.array(black_lines_xyz)
    
    # 4. Extract Starting Vertex Coordinates
    vertices = np.array(vector_poly.exterior.coords)[:-1] # Drop the closing duplicate point
    
    # 5. Iterative Greedy Edge Adjustment Loop
    print(f"Starting area-driven correction for Plot {plot_no}... Target: {target_area:,.1f} sqm")
    
    for iteration in range(max_iterations):
        current_poly = Polygon(vertices)
        current_area = current_poly.area
        initial_error = abs(current_area - target_area)
        
        if initial_error / target_area < 0.02: # Stop early if we are within 2% of the goal
            break
            
        # Determine if we need to expand or shrink the layout to hit our target size
        should_expand = current_area < target_area
        
        # Calculate distances from every vertex to all detected black boundary pixels
        spatial_distances = distance.cdist(vertices, black_lines_xyz)
        closest_pixel_indices = np.argmin(spatial_distances, axis=1)
        
        for i in range(len(vertices)):
            target_pixel_coord = black_lines_xyz[closest_pixel_indices[i]]
            
            # Create a vector pointing from current vertex to the closest image line
            direction_vector = target_pixel_coord - vertices[i]
            norm = np.linalg.norm(direction_vector)
            
            if norm == 0: continue
            unit_direction = direction_vector / norm
            
            # Propose a small test step along the alignment line
            proposed_vertex = vertices[i] + (unit_direction * step_size_meters)
            
            # Build a temporary test polygon to evaluate the change
            test_vertices = vertices.copy()
            test_vertices[i] = proposed_vertex
            test_poly = Polygon(test_vertices)
            test_area = test_poly.area
            
            # --- THE DIRECTION DIRECTION GATE ---
            # Check if this movement updates our total size in the correct direction
            is_moving_correctly = (should_expand and test_area > current_area) or (not should_expand and test_area < current_area)
            
            if is_moving_correctly:
                # Keep the move: Update this vertex coordinate position
                vertices[i] = proposed_vertex
            else:
                # Reverse the step direction to search the opposite path
                reversed_vertex = vertices[i] - (unit_direction * step_size_meters)
                test_vertices[i] = reversed_vertex
                test_poly_rev = Polygon(test_vertices)
                
                if abs(test_poly_rev.area - target_area) < initial_error:
                    vertices[i] = reversed_vertex

    # 6. Build Final Corrected Shape
    optimized_poly = Polygon(vertices)
    final_area = optimized_poly.area
    final_error_pct = abs(final_area - target_area) / target_area
    
    # 7. Compile Final Output Tracking Metadata
    if final_error_pct <= 0.05:
        payload["status"] = "corrected"
        payload["confidence"] = round(1.0 - final_error_pct, 2)
        payload["method_note"] = f"Corrected via edge optimization loop. Final area matches within {final_error_pct:.1%}."
        payload["geometry"] = mapping(optimized_poly)
    else:
        payload["status"] = "flagged"
        payload["confidence"] = round(1.0 - final_error_pct, 2) if final_error_pct < 1.0 else 0.0
        payload["method_note"] = f"Flagged: Unable to reach target area using image boundaries. Size error remains at {final_error_pct:.1%}."
        payload["geometry"] = mapping(optimized_poly)
        
    # 8. Render Visual Verification Frame
    fig, ax = plt.subplots(figsize=(8, 8))
    xmin, ymax = cropped_transform.c, cropped_transform.f
    xmax = xmin + (cropped_raster.shape[2] * cropped_transform.a)
    ymin = ymax + (cropped_raster.shape[1] * cropped_transform.e)
    
    ax.imshow(binary_contrast, cmap='gray', extent=[xmin, xmax, ymin, ymax], alpha=0.5)
    
    # Draw original vector layer vs the optimized version
    ax.plot(*raw_vector_geom.exterior.xy, color='red', linestyle='--', linewidth=1.5, label='Original Base Layer')
    ax.plot(*optimized_poly.exterior.xy, color='cyan', linewidth=2.5, label='Optimized Edge Output')
    
    # Render tracking box details
    box_text = (
        f"EDGE SEEKER AUDIT LOG\n====================\n"
        f"Plot ID        : {payload['plot_number']}\n"
        f"Registry Area  : {target_area:,.1f} sqm\n"
        f"Optimized Area : {final_area:,.1f} sqm\n"
        f"Status         : {payload['status'].upper()}\n"
        f"Confidence     : {payload['confidence']}"
    )
    b_color = "lightgreen" if payload["status"] == "corrected" else "coral"
    ax.text(0.03, 0.97, box_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
            family='monospace', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor=b_color, alpha=0.9))
    
    ax.set_title(f"Area-Constrained Edge Optimization — Plot {plot_no}", fontsize=11, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    plt.show()
    
    return payload

# optimize_edges_by_area_target("input.geojson",1)


import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.affinity import translate
from scipy.spatial import distance

def optimize_and_audit_dual_engine(geojson_path, plot_no, tiff_boundary="boundaries.tif", tiff_imagery="imagery.tif", recorded_area_col="recorded_area_sqm", image_buffer_meters=60, step_size_meters=0.5, max_iterations=30):
    """
    Fuses structural skeleton data (boundaries.tif) and intensity gradients (imagery.tif).
    Optimizes individual polygon vertices using the official recorded registry area 
    as a strict direction gate. Returns a standardized tracking payload.
    """
    # 1. Initialize Standard Output Audit Payload
    payload = {
        "plot_number": int(plot_no),
        "status": "flagged",
        "confidence": 0.0,
        "method_note": "",
        "geometry": None
    }
    
    # 2. Load Vector Layer and Target Metrics
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]].copy()
    raw_vector_geom = target_plot.geometry.values[0]
    payload["geometry"] = mapping(raw_vector_geom)  # Default fallback
    
    if recorded_area_col in target_plot.columns:
        target_area = float(target_plot[recorded_area_col].values[0])
    else:
        payload["method_note"] = f"Flagged: Missing area target data column '{recorded_area_col}' in GeoJSON registry."
        return payload
        
    if isinstance(raw_vector_geom, MultiPolygon):
        vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area)
    else:
        vector_poly = raw_vector_geom
        
    # 3. Stream and Process Primary Boundary Raster
    with rasterio.open(tiff_boundary) as src_bound:
        if target_plot.crs != src_bound.crs:
            target_plot = target_plot.to_crs(src_bound.crs)
            raw_vector_geom = target_plot.geometry.values[0]
            vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area) if isinstance(raw_vector_geom, MultiPolygon) else raw_vector_geom
            
        buffered_shape = vector_poly.buffer(image_buffer_meters)
        bound_raster, bound_transform = rasterio.mask.mask(src_bound, [buffered_shape], crop=True)
    
    # Isolate boundary lines (assume dark lines are foreground features < 128)
    bound_gray = bound_raster[0].astype(float) if bound_raster.shape[0] == 1 else (0.299 * bound_raster[0] + 0.587 * bound_raster[1] + 0.114 * bound_raster[2])
    bound_mask = (bound_gray < 128)
    
    # 4. Stream and Process Fallback Imagery Raster for Gradient Support
    from skimage.transform import resize

    # 4. Stream and Process Fallback Imagery Raster for Gradient Support
    with rasterio.open(tiff_imagery) as src_img:
        img_raster, img_transform = rasterio.mask.mask(src_img, [buffered_shape], crop=True)
        
    img_gray = img_raster[0].astype(float) if img_raster.shape[0] == 1 else (0.299 * img_raster[0] + 0.587 * img_raster[1] + 0.114 * img_raster[2])
    low_thresh = np.percentile(img_gray, 15)
    img_mask_raw = (img_gray <= low_thresh)
    
    # Resample the imagery mask to match the exact structural grid shape of the boundary raster
    img_mask = resize(img_mask_raw, bound_mask.shape, order=0, preserve_range=True, anti_aliasing=False).astype(bool)
    
    # 5. Fuse the Maps: Combine explicit boundaries with sharp imagery gradients safely!
    fused_binary_lines = np.logical_or(bound_mask, img_mask)
    
    # Extract coordinate positions of fused boundary hints
    line_rows, line_cols = np.where(fused_binary_lines == True)
    if len(line_rows) == 0:
        payload["method_note"] = "Flagged: Zero fused spatial boundary lines found in local search buffers."
        return payload
        
    # Transform pixel coordinates to ground space (Meters)
    fused_lines_xy = np.array([bound_transform * (c, r) for r, c in zip(line_rows, line_cols)])
    
    # 6. Initialize Mutable Vertex Positions
    vertices = np.array(vector_poly.exterior.coords)[:-1]
    original_vertices = vertices.copy()
    
    # 7. Coordinate Descent Loop with Strict Area Direction Gate
    for iteration in range(max_iterations):
        current_poly = Polygon(vertices)
        current_area = current_poly.area
        current_error = abs(current_area - target_area)
        
        if (current_error / target_area) < 0.01:  # Convergence threshold at 1%
            break
            
        should_expand = current_area < target_area
        
        # Matrix distance search to locate nearest physical ground landmarks
        spatial_matrix = distance.cdist(vertices, fused_lines_xy)
        closest_indices = np.argmin(spatial_matrix, axis=1)
        
        for i in range(len(vertices)):
            target_landmark = fused_lines_xy[closest_indices[i]]
            dir_vector = target_landmark - vertices[i]
            dist_norm = np.linalg.norm(dir_vector)
            
            if dist_norm == 0: continue
            unit_vector = dir_vector / dist_norm
            
            # Test step forward
            test_vertices = vertices.copy()
            test_vertices[i] = vertices[i] + (unit_vector * step_size_meters)
            test_area = Polygon(test_vertices).area
            
            if (should_expand and test_area > current_area) or (not should_expand and test_area < current_area):
                vertices[i] = test_vertices[i]
            else:
                # Test step backward if forward path fails the target gate direction
                rev_vertices = vertices.copy()
                rev_vertices[i] = vertices[i] - (unit_vector * step_size_meters)
                if abs(Polygon(rev_vertices).area - target_area) < current_error:
                    vertices[i] = rev_vertices[i]

    # 8. Post-Optimization Calculations & Audit Log Assembly
    optimized_poly = Polygon(vertices)
    final_area = optimized_poly.area
    final_area_error_pct = abs(final_area - target_area) / target_area
    
    # Compute precise baseline offset translation components
    orig_poly = Polygon(original_vertices)
    dX = optimized_poly.centroid.x - orig_poly.centroid.x
    dY = optimized_poly.centroid.y - orig_poly.centroid.y
    
    # Calculate geometric shape overlap score (IoU)
    intersection_space = optimized_poly.intersection(orig_poly).area
    union_space = optimized_poly.union(orig_poly).area
    iou_score = intersection_space / union_space if union_space > 0 else 0
    
    # 9. Formulate Final Audit State Criteria
    if final_area_error_pct <= 0.05:
        payload["status"] = "corrected"
        payload["confidence"] = round(float(1.0 - final_area_error_pct), 2)
        payload["method_note"] = f"Corrected: Fused boundary optimizer matched registry within {final_area_error_pct:.1%}."
        payload["geometry"] = mapping(optimized_poly)
    else:
        payload["status"] = "flagged"
        payload["confidence"] = round(float(iou_score), 2)
        payload["method_note"] = f"Flagged: High structural variance. Boundary layout area error remaining at {final_area_error_pct:.1%}."
        payload["geometry"] = mapping(optimized_poly)

    # 10. Generate Unified Diagnostics Quality Control Display Plot
    fig, ax = plt.subplots(figsize=(9, 9))
    xmin, ymax = bound_transform.c, bound_transform.f
    xmax = xmin + (bound_raster.shape[2] * bound_transform.a)
    ymin = ymax + (bound_raster.shape[1] * bound_transform.e)
    
    # Display the fused high-contrast digital line landscape background
    visual_fused = np.where(fused_binary_lines, 0, 255)
    ax.imshow(visual_fused, cmap='gray', extent=[xmin, xmax, ymin, ymax], alpha=0.5)
    
    # Overlay the vector configurations
    ax.plot(*orig_poly.exterior.xy, color='red', linestyle='--', linewidth=1.5, label='Original Registry Vector')
    ax.plot(*optimized_poly.exterior.xy, color='cyan', linewidth=3.0, label='Optimized Corrected Output')
    
    # Plot anchors
    ax.plot(orig_poly.centroid.x, orig_poly.centroid.y, marker='o', color='red', markersize=8, linestyle='None')
    ax.plot(optimized_poly.centroid.x, optimized_poly.centroid.y, marker='o', color='cyan', markersize=8, linestyle='None')
    
    # Compile the final spatial data panel box
    box_text = (
        f"FUSED ENGINE AUDIT LOG\n"
        f"======================\n"
        f"Plot ID        : {payload['plot_number']}\n"
        f"Status         : {payload['status'].upper()}\n"
        f"Confidence     : {payload['confidence']:.2f}\n"
        f"----------------------\n"
        f"Registry Area  : {target_area:,.1f} sqm\n"
        f"Optimized Area : {final_area:,.1f} sqm\n"
        f"Area Deviation : {final_area_error_pct:.2%}\n"
        f"Shape Overlap  : {iou_score:.2%} IoU\n"
        f"Translation Delta: [{dX:+.1f}m, {dY:+.1f}m]"
    )
    box_color = "lightgreen" if payload["status"] == "corrected" else "coral"
    ax.text(0.03, 0.97, box_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
            family='monospace', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.95))
    
    ax.set_title(f"Fused-Layer Coordinate Optimization — Plot {plot_no}", fontsize=11, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    return payload

optimize_and_audit_dual_engine("input.geojson",300)


import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.affinity import translate
from scipy.spatial import distance
from skimage.transform import resize

def optimize_and_audit_strict_gate(geojson_path, plot_no, tiff_boundary="boundaries.tif", tiff_imagery="imagery.tif", recorded_area_col="recorded_area_sqm", image_buffer_meters=60, step_size_meters=0.5, max_iterations=30, strict_ratio_tolerance=0.15):
    """
    Optimizes plot boundaries using fused rasters only if the initial area profile 
    closely matches the official registry records. Otherwise, safely flags and skips.
    """
    # 1. Initialize Output Audit Payload (Default to safe FLAGGED state)
    payload = {
        "plot_number": int(plot_no),
        "status": "flagged",
        "confidence": 0.0,
        "method_note": "",
        "geometry": None
    }
    
    # 2. Load Vector and Target Metadata
    gdf = gpd.read_file(geojson_path)
    target_plot = gdf.iloc[[plot_no]].copy()
    raw_vector_geom = target_plot.geometry.values[0]
    payload["geometry"] = mapping(raw_vector_geom)  # Default fallback geometry
    
    if recorded_area_col in target_plot.columns:
        target_area = float(target_plot[recorded_area_col].values[0])
    else:
        payload["method_note"] = "Flagged: Missing target area column in registry dataset."
        return payload
        
    if isinstance(raw_vector_geom, MultiPolygon):
        vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area)
    else:
        vector_poly = raw_vector_geom
        
    # --- THE INTELLIGENT GATEKEEPER PASS ---
    initial_vector_area = vector_poly.area
    initial_area_ratio = initial_vector_area / target_area
    
    # If the vector shape area differs from the registry by more than our tolerance, skip it immediately
    if abs(1.0 - initial_area_ratio) > strict_ratio_tolerance:
        payload["status"] = "flagged"
        payload["confidence"] = 0.0
        payload["method_note"] = f"Flagged: Skipped by Gatekeeper. Area ratio ({initial_area_ratio:.2f}) deviates too far from 1.0."
        return payload

    # 3. Stream and Process Primary Boundary Raster
    with rasterio.open(tiff_boundary) as src_bound:
        if target_plot.crs != src_bound.crs:
            target_plot = target_plot.to_crs(src_bound.crs)
            raw_vector_geom = target_plot.geometry.values[0]
            vector_poly = max(raw_vector_geom.geoms, key=lambda p: p.area) if isinstance(raw_vector_geom, MultiPolygon) else raw_vector_geom
            
        buffered_shape = vector_poly.buffer(image_buffer_meters)
        bound_raster, bound_transform = rasterio.mask.mask(src_bound, [buffered_shape], crop=True)
    
    bound_gray = bound_raster[0].astype(float) if bound_raster.shape[0] == 1 else (0.299 * bound_raster[0] + 0.587 * bound_raster[1] + 0.114 * bound_raster[2])
    bound_mask = (bound_gray < 128)
    
    # 4. Stream and Process Fallback Imagery Raster
    with rasterio.open(tiff_imagery) as src_img:
        img_raster, img_transform = rasterio.mask.mask(src_img, [buffered_shape], crop=True)
        
    img_gray = img_raster[0].astype(float) if img_raster.shape[0] == 1 else (0.299 * img_raster[0] + 0.587 * img_raster[1] + 0.114 * img_raster[2])
    low_thresh = np.percentile(img_gray, 15)
    img_mask_raw = (img_gray <= low_thresh)
    
    # Safe Resampling via skimage to prevent NumPy broadcast shape mismatches
    img_mask = resize(img_mask_raw, bound_mask.shape, order=0, preserve_range=True, anti_aliasing=False).astype(bool)
    
    # 5. Fuse Structural Data and Gradients
    fused_binary_lines = np.logical_or(bound_mask, img_mask)
    line_rows, line_cols = np.where(fused_binary_lines == True)
    
    if len(line_rows) == 0:
        payload["method_note"] = "Flagged: No clear line features found in the local raster subsets."
        return payload
        
    fused_lines_xy = np.array([bound_transform * (c, r) for r, c in zip(line_rows, line_cols)])
    
    # 6. Initialize Vertex Positions
    vertices = np.array(vector_poly.exterior.coords)[:-1]
    original_vertices = vertices.copy()
    
    # 7. Coordinate Descent Loop with Strict Area Compass
    for iteration in range(max_iterations):
        current_poly = Polygon(vertices)
        current_area = current_poly.area
        current_error = abs(current_area - target_area)
        
        if (current_error / target_area) < 0.01:
            break
            
        should_expand = current_area < target_area
        spatial_matrix = distance.cdist(vertices, fused_lines_xy)
        closest_indices = np.argmin(spatial_matrix, axis=1)
        
        for i in range(len(vertices)):
            target_landmark = fused_lines_xy[closest_indices[i]]
            dir_vector = target_landmark - vertices[i]
            dist_norm = np.linalg.norm(dir_vector)
            
            if dist_norm == 0: continue
            unit_vector = dir_vector / dist_norm
            
            test_vertices = vertices.copy()
            test_vertices[i] = vertices[i] + (unit_vector * step_size_meters)
            test_area = Polygon(test_vertices).area
            
            if (should_expand and test_area > current_area) or (not should_expand and test_area < current_area):
                vertices[i] = test_vertices[i]
            else:
                rev_vertices = vertices.copy()
                rev_vertices[i] = vertices[i] - (unit_vector * step_size_meters)
                if abs(Polygon(rev_vertices).area - target_area) < current_error:
                    vertices[i] = rev_vertices[i]

    # 8. Post-Optimization Calibration Check
    optimized_poly = Polygon(vertices)
    final_area = optimized_poly.area
    final_area_error_pct = abs(final_area - target_area) / target_area
    
    orig_poly = Polygon(original_vertices)
    dX = optimized_poly.centroid.x - orig_poly.centroid.x
    dY = optimized_poly.centroid.y - orig_poly.centroid.y
    
    intersection_space = optimized_poly.intersection(orig_poly).area
    union_space = optimized_poly.union(orig_poly).area
    iou_score = intersection_space / union_space if union_space > 0 else 0
    
    # 9. Final Quality Check Gate: Only mark corrected if final error is under 5%
    if final_area_error_pct <= 0.05:
        payload["status"] = "corrected"
        payload["confidence"] = round(float(1.0 - final_area_error_pct), 2)
        payload["method_note"] = f"Corrected: Successfully optimized. Area error within {final_area_error_pct:.2%}."
        payload["geometry"] = mapping(optimized_poly)
    else:
        payload["status"] = "flagged"
        payload["confidence"] = round(float(iou_score), 2)
        payload["method_note"] = f"Flagged: Optimization failed to converge within 5% area threshold. Final error: {final_area_error_pct:.2%}."
        payload["geometry"] = mapping(orig_poly) # Reset to original to prevent distortions

    # 10. Generate Quality Control Display Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    xmin, ymax = bound_transform.c, bound_transform.f
    xmax = xmin + (bound_raster.shape[2] * bound_transform.a)
    ymin = ymax + (bound_raster.shape[1] * bound_transform.e)
    
    visual_fused = np.where(fused_binary_lines, 0, 255)
    ax.imshow(visual_fused, cmap='gray', extent=[xmin, xmax, ymin, ymax], alpha=0.5)
    
    ax.plot(*orig_poly.exterior.xy, color='red', linestyle='--', linewidth=1.5, label='Original Input')
    if payload["status"] == "corrected":
        ax.plot(*optimized_poly.exterior.xy, color='cyan', linewidth=3.0, label='Corrected Output')
        ax.plot(optimized_poly.centroid.x, optimized_poly.centroid.y, marker='o', color='cyan', markersize=8)
    
    ax.plot(orig_poly.centroid.x, orig_poly.centroid.y, marker='o', color='red', markersize=8)
    
    box_text = (
        f"STRICT GATEKEEPER AUDIT LOG\n"
        f"===========================\n"
        f"Plot ID        : {payload['plot_number']}\n"
        f"Status         : {payload['status'].upper()}\n"
        f"Confidence     : {payload['confidence']:.2f}\n"
        f"---------------------------\n"
        f"Registry Area  : {target_area:,.1f} sqm\n"
        f"Optimized Area : {final_area:,.1f} sqm\n"
        f"Area Ratio     : {final_area / target_area:.2f}\n"
        f"Shape Overlap  : {iou_score:.2%} IoU"
    )
    box_color = "lightgreen" if payload["status"] == "corrected" else "coral"
    ax.text(0.03, 0.97, box_text, transform=ax.transAxes, fontsize=9, fontweight='bold',
            family='monospace', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor=box_color, alpha=0.95))
    
    ax.set_title(f"Strict-Gated Spatial Optimization — Plot {plot_no}", fontsize=11, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    return payload








import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import json

def run_pipeline_and_export_geojson(input_geojson_path, output_geojson_path, from_plot,total_plots):
    """
    Runs the optimization function across all plots, accumulates the standard 
    audit tracking payloads, and exports a clean, unified GeoJSON file.
    """
    all_payloads = []
    
    print(f"Starting processing pipeline for {total_plots} plots...")
    
    for i in range(from_plot,total_plots):
        try:
            # Execute your optimized dual-engine strict gate function
            # (Make sure this matches the exact name of the function in your file)
            payload = optimize_and_audit_strict_gate(input_geojson_path, plot_no=i)
            all_payloads.append(payload)
            print(f"Processed Plot {i}: Status = {payload['status'].upper()}")
        except Exception as e:
            print(f"Fatal error processing plot index {i}: {str(e)}")
            
    # --- CONVERT PAYLOADS TO GEOPANDAS DATASTRUCTURE ---
    features = []
    for item in all_payloads:
        # Reconstruct the Shapely geometric feature from the dictionary mapping
        geom_obj = shape(item["geometry"])
        
        # Structure the row attributes exactly to your standards
        attributes = {
            "plot_number": int(item["plot_number"]),
            "status": str(item["status"]),       # 'corrected' or 'flagged'
            "confidence": float(item["confidence"]), # 0.0 to 1.0 based on precision
            "method_note": str(item["method_note"])  # Audit trail reasons
        }
        
        # Build an individual GeoDataFrame record row
        gdf_row = gpd.GeoDataFrame([attributes], geometry=[geom_obj])
        features.append(gdf_row)
        
    # Merge all individual rows into one master dataset
    output_gdf = pd.concat(features, ignore_index=True)
    
    # 2. Inherit CRS Coordinate reference system from original input map
    original_gdf = gpd.read_file(input_geojson_path)
    output_gdf.crs = original_gdf.crs
    
    # 3. Write securely to disk as a standard GeoJSON file
    output_gdf.to_file(output_geojson_path, driver="GeoJSON")
    print(f"\nSuccessfully generated and saved unified audit map to: {output_geojson_path}")
