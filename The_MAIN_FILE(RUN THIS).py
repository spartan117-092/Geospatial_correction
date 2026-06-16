import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

# ==========================================================================================================================
# 1. DATA EXTRACTION LAYER
# ==========================================================================================================================
def extract_raster_features(polygon, raster_src, image_buffer_meters=60):
    """
    Isolates a local AOI from a TIFF image via a buffered polygon mask,
    converts it to a binary line-mask, and converts pixel coordinates
    directly into global CRS map coordinates (XY).
    """
    buffered_shape = polygon.buffer(image_buffer_meters)

    try:
        bound_raster, bound_transform = rasterio.mask.mask(
            raster_src,
            [buffered_shape],
            crop=True
        )
    except ValueError:
        return None

    if bound_raster.shape[0] == 1:
        bound_gray = bound_raster[0].astype(float)
    else:
        bound_gray = (
            0.299 * bound_raster[0] +
            0.587 * bound_raster[1] +
            0.114 * bound_raster[2]
        ).astype(float)

    fused_binary_lines = bound_gray > 128
    rows, cols = np.where(fused_binary_lines)

    if len(rows) == 0:
        return None

    # Vectorized fast translation
    xs = bound_transform.a * cols + bound_transform.b * rows + bound_transform.c
    ys = bound_transform.d * cols + bound_transform.e * rows + bound_transform.f

    return np.column_stack((xs, ys))
# ==========================================================================================================================
# ==========================================================================================================================
def apply_global_consensus_shift(vertices, edge_shifts, edge_normals):
    """
    Computes a rigid-body translation based on mean consensus of edge votes.
    Returns: shifted_vertices, net_translation
    """
    all_vectors = [shift * normal for shift, normal in zip(edge_shifts, edge_normals) if abs(shift) > 0.1]
    
    if len(all_vectors) > 0:
        net_translation = np.mean(all_vectors, axis=0)
        return vertices + net_translation, net_translation
    
    return vertices, np.array([0.0, 0.0])



# ==========================================================================================================================
# 2. MATHEMATICAL TRANSFORMS & HELPERS
# ==========================================================================================================================
def compute_local_normals_and_tangents(vertices):
    """Calculates perpendicular outward normals and tangents for structural vertices."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    tangents = np.zeros((n, 2))
    
    for i in range(n):
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        
        v_prev = vertices[prev_idx] - vertices[i]
        v_next = vertices[next_idx] - vertices[i]
        
        len_prev = np.linalg.norm(v_prev)
        len_next = np.linalg.norm(v_next)
        
        u_prev = v_prev / len_prev if len_prev > 0 else v_prev
        u_next = v_next / len_next if len_next > 0 else v_next
        
        bisector = u_prev + u_next
        len_bi = np.linalg.norm(bisector)
        
        if len_bi < 1e-4:
            u_normal = np.array([-u_prev[1], u_prev[0]])
        else:
            u_normal = -bisector / len_bi
            
        u_tangent = np.array([-u_normal[1], u_normal[0]])
        
        normals[i] = u_normal
        tangents[i] = u_tangent
        
    return normals, tangents


def estimate_raster_tangent_pca(center_coord, all_raster_xy, window_radius=10.0):
    """Uses PCA on local neighborhood pixels to find the local heading of the road feature."""
    deltas = all_raster_xy - center_coord
    distances = np.linalg.norm(deltas, axis=1)
    local_points = all_raster_xy[distances <= window_radius]
    
    if len(local_points) < 3:
        return None
        
    mean_centered = local_points - np.mean(local_points, axis=0)
    cov_matrix = np.cov(mean_centered, rowvar=False)
    
    if cov_matrix.ndim < 2 or np.isnan(cov_matrix).any():
        return None
        
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    dominant_direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    return dominant_direction / np.linalg.norm(dominant_direction)
def calculate_vertex_overlap_percentage(vertices, raster_xy, hit_tolerance_meters=2.0):
    """Calculates what percentage of vertices sit within an acceptable proximity buffer."""
    hits = 0
    for v in vertices:
        distances = np.linalg.norm(raster_xy - v, axis=1)
        if np.any(distances <= hit_tolerance_meters):
            hits += 1
    return (hits / len(vertices)) * 100.0


# ========================================================================================================================
def get_edge_data(vertices, fused_lines_xy, config,visualization_payload=None):
    edge_shifts = []
    edge_normals = []
    num_vertices = len(vertices)
    
    # Extracting parameters from config (ensure these match your main function)
    snap_search_radius_meters = config.get('snap_search_radius_meters', 2.0)
    edge_sampling_step_meters = config.get('edge_sampling_step_meters', 1.0)
    angle_thresh_rad = config.get('angle_thresh_rad', 0.5)

    for i in range(num_vertices):
        v_start = vertices[i]
        v_end = vertices[(i + 1) % num_vertices]
        edge_vector = v_end - v_start
        edge_length = np.linalg.norm(edge_vector)
        if visualization_payload is not None:
            # We must use 'i' to update the specific edge
            visualization_payload["edge_shifts"][i] = median_shift
        
        if edge_length < 1e-3:
            edge_shifts.append(0.0)
            edge_normals.append(np.array([0.0, 0.0]))
            continue

        u_tangent = edge_vector / edge_length
        u_normal = np.array([u_tangent[1], -u_tangent[0]])
        sample_distances = np.arange(0.0, edge_length, edge_sampling_step_meters)
        
        local_edge_votes = []
        for step_dist in sample_distances:
            virtual_pt = v_start + (step_dist * u_tangent)
            deltas = fused_lines_xy - virtual_pt
            proj_n = np.dot(deltas, u_normal)
            proj_t = np.dot(deltas, u_tangent)
            mask = (np.abs(proj_n) <= snap_search_radius_meters) & (np.abs(proj_t) <= 5.0)
            
            if np.any(mask):
                valid_steps = proj_n[mask]
                local_edge_votes.append(float(np.mean(valid_steps)))

        median_shift = float(np.median(local_edge_votes)) if local_edge_votes else 0.0
        edge_shifts.append(median_shift)
        edge_normals.append(u_normal)
        
    return edge_shifts, edge_normals

def get_global_translation_vector(vertices, fused_lines_xy, config):
    """
    Computes a single rigid shift (dx, dy) for the entire polygon.
    """
    edge_shifts, edge_normals = get_edge_data(vertices, fused_lines_xy, config)
    
    # Calculate vector = distance * direction
    translation_votes = []
    for shift, normal in zip(edge_shifts, edge_normals):
        if abs(shift) > 0.1:  # Filter out noise
            translation_votes.append(shift * normal)
            
    if not translation_votes:
        return np.array([0.0, 0.0])
        
    # Return the mean of all votes
    return np.mean(translation_votes, axis=0)

def optimize_simple_polygon(vertices, fused_lines_xy, target_area, config):
    """
    Specialized optimizer for 4-5 sided polygons.
    Performs a rigid body translation based on edge consensus.
    """

    visualization_payload = {
        "virtual_pts": [], "search_beams": [], "matched_pixels": [], "edge_shifts": {}
    }
    # 1. Calculate 'Before' metrics
    # Ensure this matches your existing metrics calculation logic
    overlap_before = calculate_vertex_overlap_percentage(vertices, fused_lines_xy)
    
    # 2. Get the global translation vector (The "Brain" of the operation)
    # This calls get_edge_data internally to scan the raster
    net_translation = get_global_translation_vector(vertices, fused_lines_xy, config)
    
    # 3. Apply the rigid shift to all vertices
    # This moves the polygon as one single piece without warping
    updated_vertices = vertices + net_translation
    
    # 4. Finalize the geometry
    poly_shifted = Polygon(np.vstack([updated_vertices, updated_vertices[0]]))
    
    # 5. Calculate 'After' metrics
    overlap_after = calculate_vertex_overlap_percentage(updated_vertices, fused_lines_xy)
    
    area_after = poly_shifted.area
    # area_deficit_after_pct should be a percentage (0-100)
    area_deficit_after_pct = (1.0 - (area_after / target_area)) * 100.0 if target_area > 0 else 0.0

    # 6. Return the full payload for the GeoJSON
    return {
        "vertices": np.vstack([updated_vertices, updated_vertices[0]]),
        "polygon": poly_shifted,
        "overlap_before": overlap_before,
        "overlap_after": overlap_after,
        "area_deficit_before_pct": 0.0, # Usually captured before the call
        "area_deficit_after_pct": area_deficit_after_pct,
        "rolled_back": False,
        "status_gate": "Simple rigid shift optimized",
        "viz_data": visualization_payload,
    }
# =======================================================================================================================

# ==========================================================================================================================
# 3. PURE GEOMETRIC OPTIMIZER
# ==========================================================================================================================
def should_fallback(vertices, target_area, base_truth_anchor=None, max_move=26.0) -> tuple[bool, str]:
    """
    Unified gatekeeper for geometric stability. 
    Handles initial complexity filtration AND live translation breach detection.
    """
    # --- PHASE 1: INITIAL COMPLEXITY FILTRATION ---
    if base_truth_anchor is None:
        num_vertices = len(vertices)
        if num_vertices > 15:
            return True, f"High vertex complexity count ({num_vertices} vertices)"
            
        poly = Polygon(np.vstack([vertices, vertices[0]]))
        if target_area > 0:
            area_ratio = poly.area / target_area
            if area_ratio > 2.0 or area_ratio < 0.5:
                return True, f"Dangerous initial area ratio ({area_ratio:.2f}x)"
                
        return False, "SUCCESFULY CORRECTED"

    # --- PHASE 2: LIVE MOTION BREACH DETECTION (Mid-Loop Check) ---
    for i in range(len(vertices)):
        displacement = vertices[i] - base_truth_anchor[i]
        distance = float(np.linalg.norm(displacement))
        
        if distance > max_move:
            return True, f"Vertex {i} breached max translation threshold ({distance:.2f}m > {max_move}m)"
            
    return False, ""


def optimize_vertices(vertices, fused_lines_xy, target_area, config=None):
    """
    Edge-Centric Optimization System.
    Uses an external modular fallback engine to safely filter complex shapes.
    """
    if config is None:
        config = {}

    max_iterations = config.get("max_iterations", 5)
    snap_search_radius_meters = config.get("snap_search_radius_meters", 15.0)
    edge_sampling_step_meters = config.get("edge_sampling_step_meters", 1.5)
    orientation_angle_thresh_deg = config.get("orientation_angle_thresh_deg", 25.0)
    angle_thresh_rad = np.radians(orientation_angle_thresh_deg)
    
    corner_angle_threshold_deg = config.get("corner_angle_threshold_deg", 110.0)
    max_vertex_move_meters = float(config.get("max_vertex_move_meters", 60.0))

    if np.allclose(vertices[0], vertices[-1]):
        vertices = vertices[:-1].copy()
    else:
        vertices = vertices.copy()

    original_vertices = vertices.copy()
    num_vertices = len(vertices)
    visualization_payload = {
        "virtual_pts": [], "search_beams": [], "matched_pixels": [], "edge_shifts": {}
    }

    # ==========================================================================
    # CALCULATE BEFORE METRICS
    # ==========================================================================
    poly_before = Polygon(np.vstack([original_vertices, original_vertices[0]]))
    area_before = poly_before.area
    
    overlap_before = calculate_vertex_overlap_percentage(original_vertices, fused_lines_xy, hit_tolerance_meters=2.0)
    area_deficit_before = abs(area_before - target_area)


        
  
    # num_vertices = len(vertices)

    # --- THE BYPASS: If 3 vertices, return the original data immediately ---
    if num_vertices <= 3:
        print(f"Bypassing optimization for {num_vertices}-sided polygon (Triangle/Degenerate)...")
        poly_before = Polygon(np.vstack([vertices, vertices[0]]))
        return {
            "vertices": np.vstack([vertices, vertices[0]]),
            "polygon": poly_before,
            "overlap_before": float(calculate_vertex_overlap_percentage(vertices, fused_lines_xy)),
            "overlap_after": float(calculate_vertex_overlap_percentage(vertices, fused_lines_xy)),
            "area_deficit_before_pct": 0.0,
            "area_deficit_after_pct": 0.0,
            "rolled_back": True,  # Flagged as 'rolled_back' so the status is recorded correctly
            "status_gate": "Bypassed: Triangular/Degenerate geometry",
            "viz_data": {"virtual_pts": [], "search_beams": [], "matched_pixels": [], "edge_shifts": {}}
        }
    print(f"the number side={num_vertices}")
    # =================================================================================
    if num_vertices<=7 and num_vertices>=4:
        print(f"Applying rigid centroid shift...")
        result = optimize_simple_polygon(vertices, fused_lines_xy, target_area, config)
        # Return immediately so you don't run the complex logic twice
        return result
    # =================================================================================




    print("Sides More than 7...")
    # ==========================================================================
    # MODULAR DETACHED FALLBACK ENGINE TRIGGER
    # ==========================================================================
    trigger_fallback, fallback_reason = should_fallback(original_vertices, target_area)
    
    if trigger_fallback:
        initial_deficit_pct = (area_deficit_before / target_area) * 100.0 if target_area > 0 else 0.0
        
        return {
            "vertices": np.vstack([original_vertices, original_vertices[0]]),
            "polygon": poly_before,
            "overlap_before": float(overlap_before),
            "overlap_after": float(overlap_before),
            "area_deficit_before_pct": float(initial_deficit_pct),
            "area_deficit_after_pct": float(initial_deficit_pct),
            "rolled_back": True,
            "status_gate": f"Fallback triggered: {fallback_reason}",
            "viz_data": visualization_payload
        }

    rolled_back = False
    status_msg = "Optimized successfully"

    # ==========================================================================
    # CORE OPTIMIZATION ITERATION LOOP
    # ==========================================================================
    for iteration in range(max_iterations):
        edge_shifts = []
        edge_normals = []
        edge_anchors = []

        # STAGE 1: SCAN EDGES AND COLLECT CONSENSUS VOTES
        for i in range(num_vertices):
            v_start = vertices[i]
            v_end = vertices[(i + 1) % num_vertices]

            edge_vector = v_end - v_start
            edge_length = np.linalg.norm(edge_vector)
            if edge_length < 1e-3:
                continue

            u_tangent = edge_vector / edge_length
            u_normal = np.array([u_tangent[1], -u_tangent[0]])

            sample_distances = np.arange(0.0, edge_length, edge_sampling_step_meters)
            if len(sample_distances) == 0:
                sample_distances = np.array([0.0])

            local_edge_votes = []

            for step_dist in sample_distances:
                virtual_pt = v_start + (step_dist * u_tangent)
                deltas = fused_lines_xy - virtual_pt
                proj_n = np.dot(deltas, u_normal)
                proj_t = np.dot(deltas, u_tangent)

                mask = (np.abs(proj_n) <= snap_search_radius_meters) & (np.abs(proj_t) <= 5.0)
                candidates = fused_lines_xy[mask]
                candidate_proj_n = proj_n[mask]

                if len(candidates) == 0:
                    continue

                valid_coords = []
                valid_steps = []

                for idx, c_coord in enumerate(candidates):
                    r_tangent = estimate_raster_tangent_pca(c_coord, fused_lines_xy, window_radius=10.0)
                    if r_tangent is None:
                        valid_coords.append(c_coord)
                        valid_steps.append(candidate_proj_n[idx])
                    elif abs(np.dot(u_tangent, r_tangent)) >= np.cos(angle_thresh_rad):
                        valid_coords.append(c_coord)
                        valid_steps.append(candidate_proj_n[idx])

                if len(valid_coords) == 0:
                    continue

                valid_coords = np.array(valid_coords)
                valid_steps = np.array(valid_steps)
                closest_idx = np.argmin(np.linalg.norm(valid_coords - virtual_pt, axis=1))

                chosen_pixel = valid_coords[closest_idx]
                chosen_shift = float(valid_steps[closest_idx])
                local_edge_votes.append(chosen_shift)

                if iteration == 0:
                    visualization_payload["virtual_pts"].append(virtual_pt)
                    visualization_payload["search_beams"].append((virtual_pt, virtual_pt + (chosen_shift * u_normal)))
                    visualization_payload["matched_pixels"].append(chosen_pixel)

            if len(local_edge_votes) > 0:
                median_edge_shift = float(np.median(local_edge_votes))
                edge_shifts.append(median_edge_shift)
                edge_normals.append(u_normal)
                edge_anchors.append(v_start)
                if iteration == 0:
                    visualization_payload["edge_shifts"][i] = median_edge_shift
            else:
                edge_shifts.append(0.0)
                edge_normals.append(u_normal)
                edge_anchors.append(v_start)
                if iteration == 0:
                    visualization_payload["edge_shifts"][i] = 0.0

        # print("\nEDGE SHIFTS")
        # for i, shift in enumerate(edge_shifts):
        #     print(f"Edge {i}: {shift:.2f}m")
        for i in range(num_vertices):
            prev_i = (i - 1) % num_vertices

            # print(
            #     f"Vertex {i}",
            #     f"PrevEdge={edge_shifts[prev_i]:.2f}",
            #     f"CurrEdge={edge_shifts[i]:.2f}",
            #     f"Difference={abs(edge_shifts[i]-edge_shifts[prev_i]):.2f}"
            # )

        # STAGE 2: ADAPTIVE SHIFTING WITH UNIFIED FALLBACK GATE
        updated_vertices = np.zeros_like(vertices)

        if iteration == 0:
            base_truth_anchor = original_vertices.copy()

        for i in range(num_vertices):
            prev_idx = (i - 1) % num_vertices
            next_idx = (i + 1) % num_vertices
            
            n_prev, d_prev = edge_normals[prev_idx], edge_shifts[prev_idx]
            n_curr, d_curr = edge_normals[i], edge_shifts[i]

            v1 = vertices[i] - vertices[prev_idx]
            v2 = vertices[next_idx] - vertices[i]
            len_v1, len_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
            
            if len_v1 > 1e-3 and len_v2 > 1e-3:
                cos_theta = np.clip(np.dot(v1, v2) / (len_v1 * len_v2), -1.0, 1.0)
                interior_angle = 180.0 - np.degrees(np.arccos(cos_theta))
            else:
                interior_angle = 180.0

            if interior_angle < corner_angle_threshold_deg:
                fully_shifted_pt = vertices[i] + (n_prev * d_prev) + (n_curr * d_curr)
            else:
                fully_shifted_pt = vertices[i] + (n_curr * d_curr)

            updated_vertices[i] = fully_shifted_pt

        # LIVE MOTION BREACH CHECK
        motion_breach, breach_reason = should_fallback(
            updated_vertices, target_area, base_truth_anchor, max_vertex_move_meters
        )
        
        if motion_breach:
            vertices = original_vertices.copy()
            rolled_back = True
            status_msg = f"Aborted: {breach_reason}"
            break


        # Crisscross Validity Check
        trial_poly = Polygon(np.vstack([updated_vertices, updated_vertices[0]]))
        if not trial_poly.is_valid:
            vertices = original_vertices.copy()
            rolled_back = True
            status_msg = "Aborted: Topologically invalid self-intersection layout generated"
            break

        position_delta = np.mean(np.linalg.norm(updated_vertices - vertices, axis=1))
        vertices = updated_vertices.copy()
        if position_delta < 0.01:
            break

# # ====================================EDGE BASED CORRECTION===========================================
#     # STAGE 2: EDGE-OFFSET RECONSTRUCTION
#         updated_vertices = np.zeros_like(vertices)
#         offset_lines = []

#         # 1. Create the offset lines for every edge
#         for i in range(num_vertices):
#             v_start = vertices[i]
#             v_end = vertices[(i + 1) % num_vertices]
#             u_normal = edge_normals[i]
#             shift = edge_shifts[i]
            
#             # Create the offset segment
#             o_start = v_start + (u_normal * shift)
#             o_end = v_end + (u_normal * shift)
#             offset_lines.append((o_start, o_end))

#         # 2. Reconstruct vertices by intersecting adjacent offset lines
#         for i in range(num_vertices):
#             prev_line = offset_lines[(i - 1) % num_vertices]
#             curr_line = offset_lines[i]
            
#             # Intersection of line (p1, p2) and (p3, p4)
#             p1, p2 = prev_line
#             p3, p4 = curr_line
            
#             # Line coefficients: Ax + By = C
#             A1, B1 = p2[1] - p1[1], p1[0] - p2[0]
#             C1 = A1 * p1[0] + B1 * p1[1]
            
#             A2, B2 = p4[1] - p3[1], p3[0] - p4[0]
#             C2 = A2 * p3[0] + B2 * p3[1]
            
#             det = A1 * B2 - A2 * B1

#             # If det is near 0, lines are parallel
#             if abs(det) < 1e-4:
#                 updated_vertices[i] = vertices[i]
#             else:
#                 x = (B2 * C1 - B1 * C2) / det
#                 y = (A1 * C2 - A2 * C1) / det
#                 new_pt = np.array([x, y])
 

#                 # 3. Safety Gate: Constrain cumulative movement
#                 dist = np.linalg.norm(new_pt - original_vertices[i])
#                 if dist > max_vertex_move_meters:
#                     # Scale back to max boundary if movement is too large
#                     updated_vertices[i] = original_vertices[i] + (new_pt - original_vertices[i]) * (max_vertex_move_meters / dist)
#                 else:
#                     updated_vertices[i] = new_pt
#                 print(f"new_pt={new_pt},original_vertices={original_vertices},maxvertexmove/dist={max_vertex_move_meters / dist}")


        # # LIVE MOTION BREACH CHECK
        # motion_breach, breach_reason = should_fallback(
        #     updated_vertices, target_area, base_truth_anchor, max_vertex_move_meters
        # )
        
        # if motion_breach:
        #     vertices = original_vertices.copy()
        #     rolled_back = True
        #     status_msg = f"Aborted: {breach_reason}"
        #     break


        # Crisscross Validity Check
        # trial_poly = Polygon(np.vstack([updated_vertices, updated_vertices[0]]))
        # if not trial_poly.is_valid:
        #     vertices = original_vertices.copy()
        #     rolled_back = True
        #     status_msg = "Aborted: Topologically invalid self-intersection layout generated"
        #     break

        position_delta = np.mean(np.linalg.norm(updated_vertices - vertices, axis=1))
        vertices = updated_vertices.copy()
        if position_delta < 0.01:
            break
        # Now proceed with your existing Validity Gate and Fallback system


    # STAGE 3: POST-PROCESSING & FINAL RETRIEVAL
    if not rolled_back:
        overlap_after = calculate_vertex_overlap_percentage(vertices, fused_lines_xy, hit_tolerance_meters=2.0)
        if overlap_after < overlap_before and overlap_before >= 35.0:
            vertices = original_vertices.copy()
            overlap_after = overlap_before
            rolled_back = True
            status_msg = "Aborted: Shift decreased boundary overlap score"
    else:
        overlap_after = overlap_before

    # GUARANTEED % METRICS CALCULATION
    final_poly = Polygon(np.vstack([vertices, vertices[0]]))
    area_after = final_poly.area
    
    area_deficit_after = abs(area_after - target_area)
    
    area_deficit_before_pct = (area_deficit_before / target_area) * 100.0 if target_area > 0 else 0.0
    area_deficit_after_pct = (area_deficit_after / target_area) * 100.0 if target_area > 0 else 0.0
    

    # ---------------------------
    return {
        "vertices": np.vstack([vertices, vertices[0]]),
        "polygon": final_poly,
        "overlap_before": float(overlap_before),
        "overlap_after": float(overlap_after),
        "area_deficit_before_pct": float(area_deficit_before_pct),
        "area_deficit_after_pct": float(area_deficit_after_pct),
        "rolled_back": rolled_back,
        "status_gate": status_msg,
        "viz_data": visualization_payload
    }


# ==========================================================================================================================
# 4. DIAGNOSTIC VISUALIZATION ENGINE (Extracted outside __main__ to avoid scope issues)
# ==========================================================================================================================
def plot_edge_voting_diagnostics(original_polygon, corrected_polygon, fused_lines_xy, viz_data, config=None):
    """
    Renders the tracking paths, search envelopes, and target pixel locks,
    along with clear displacement vectors showing exactly how each vertex 
    moves, expands, or shrinks, tagged with their precise compass directions.
    """
    import matplotlib.patches as patches

    if config is None:
        config = {}

    snap_search_radius_meters = config.get("snap_search_radius_meters", 15.0)

    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Background Feature Cloud
    if fused_lines_xy is not None:
        ax.scatter(fused_lines_xy[:, 0], fused_lines_xy[:, 1], s=1.5, color='gray', alpha=0.3, label="Raster Pixels")

    # 2. Draw Vector Rays, Virtual Centers, and Target Locks
    v_pts = np.array(viz_data["virtual_pts"])
    m_pixels = np.array(viz_data["matched_pixels"])
    beams = viz_data["search_beams"]

    # Render search radius envelopes uniformly everywhere
    has_drawn_circle = False
    for pt in v_pts:
        circle_lbl = f"Search Radius Envelopes ({snap_search_radius_meters}m)" if not has_drawn_circle else ""
        has_drawn_circle = True
        c = patches.Circle((pt[0], pt[1]), radius=snap_search_radius_meters, edgecolor='crimson', 
                           facecolor='none', linestyle='--', linewidth=0.4, alpha=0.1, zorder=2, label=circle_lbl)
        ax.add_patch(c)

    # Draw the directional normal beam paths
    has_drawn_beam = False
    for beam in beams:
        beam_lbl = "Normal Vector Search Beam" if not has_drawn_beam else ""
        has_drawn_beam = True
        ax.plot([beam[0][0], beam[1][0]], [beam[0][1], beam[1][1]], color='orange', linestyle=':', linewidth=0.8, alpha=0.6, zorder=3, label=beam_lbl)

    # Highlight target locking locations
    if len(m_pixels) > 0:
        ax.scatter(m_pixels[:, 0], m_pixels[:, 1], color='gold', s=10, zorder=4, label="Matched Road Pixel")
    if len(v_pts) > 0:
        ax.scatter(v_pts[:, 0], v_pts[:, 1], color='blue', s=6, alpha=0.5, zorder=4, label="Edge Voting Virtual Point")

    # 3. Polygon Overlays
    orig = np.array(original_polygon.exterior.coords)
    corr = np.array(corrected_polygon.exterior.coords)

    ax.plot(orig[:, 0], orig[:, 1], '--', color='red', linewidth=2, label='Original Position', zorder=5)
    ax.scatter(orig[:-1, 0], orig[:-1, 1], color='crimson', s=45, zorder=6)

    ax.plot(corr[:, 0], corr[:, 1], color='cyan', linewidth=2, label='Aligned Output', zorder=5)
    ax.scatter(corr[:-1, 0], corr[:-1, 1], color='darkcyan', s=50, zorder=6)

    # 4. COMPASS DIRECTION MAPPING ENGINE
    poly_center = np.mean(orig[:-1], axis=0)

    def get_compass_direction(pt, center):
        dx = pt[0] - center[0]
        dy = pt[1] - center[1]
        angle = np.degrees(np.arctan2(dy, dx)) % 360.0
        
        if 22.5 <= angle < 67.5:    return "North-East"
        elif 67.5 <= angle < 112.5:  return "North"
        elif 112.5 <= angle < 157.5: return "North-West"
        elif 157.5 <= angle < 202.5: return "West"
        elif 202.5 <= angle < 247.5: return "South-West"
        elif 247.5 <= angle < 292.5: return "South"
        elif 292.5 <= angle < 337.5: return "South-East"
        else:                        return "East"

    # 5. ADD VERTEX DISPLACEMENT VECTORS & COMPASS ORIENTATION TEXT
    num_vertices = len(orig) - 1
    has_drawn_arrow = False
    
    for i in range(num_vertices):
        p_start = orig[i]
        p_end = corr[i]
        
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        move_distance = np.sqrt(dx**2 + dy**2)
        
        direction_name = get_compass_direction(p_start, poly_center)
        arrow_lbl = "Vertex Shift Vector" if not has_drawn_arrow else ""
        has_drawn_arrow = True
        
        ax.annotate(
            "", 
            xy=(p_end[0], p_end[1]), 
            xytext=(p_start[0], p_start[1]),
            arrowprops=dict(
                arrowstyle="->", 
                color="lime", 
                lw=2.5, 
                mutation_scale=15,
                zorder=7
            ),
            label=arrow_lbl
        )
        
        ax.text(
            p_start[0] + dx/2 + 0.3, 
            p_start[1] + dy/2 + 0.3, 
            f"[V{i}: {direction_name}] {move_distance:.2f}m", 
            color="white", 
            fontsize=9, 
            weight="bold",
            bbox=dict(facecolor='black', alpha=0.8, edgecolor='lime', boxstyle='round,pad=0.3'),
            zorder=8
        )

    ax.set_aspect('equal')
    ax.set_title("Edge-Centric Alignment: Vector Rays, Locks & Synchronized Corner Shifts")
    ax.legend(loc='lower right')
    plt.show()


def calculate_calibrated_confidence(result) -> float:
    """
    Confidence based on final alignment quality,
    not improvement magnitude.
    """

    if result.get("rolled_back", False):
        return 0.10

    overlap_after = result.get("overlap_after", 0.0)
    translation_consensus = result.get("translation_consensus", 0.0)
    area_error_pct = result.get("area_error_pct", 100.0)
    residual_error = result.get("residual_error", 999.0)

    # ------------------------------------------------------------------
    # Alignment Score
    # ------------------------------------------------------------------
    overlap_score = overlap_after / 100.0

    # ------------------------------------------------------------------
    # Vote Agreement Score
    # ------------------------------------------------------------------
    consensus_score = np.clip(translation_consensus, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Area Preservation Score
    # Full score at 0% error
    # Falls to 0 at 20% error
    # ------------------------------------------------------------------
    area_score = max(
        0.0,
        1.0 - (area_error_pct / 20.0)
    )

    # ------------------------------------------------------------------
    # Residual Snap Quality
    # Full score below 0.5 m
    # Falls to 0 at 5 m
    # ------------------------------------------------------------------
    residual_score = max(
        0.0,
        1.0 - (residual_error / 5.0)
    )

    # ------------------------------------------------------------------
    # Weighted Fusion
    # ------------------------------------------------------------------
    confidence = (
        0.45 * overlap_score +
        0.25 * consensus_score +
        0.20 * area_score +
        0.10 * residual_score
    )

    return float(np.clip(confidence, 0.05, 0.95))




def process_plot(plot_number, polygon, target_area, raster_src, config=None):
    """Manages complete data flow pipeline execution for a single plot."""
    fused_lines_xy = extract_raster_features(polygon=polygon, raster_src=raster_src, image_buffer_meters=60)

    if fused_lines_xy is None:
        return {
            "plot_number": plot_number,
            "status": "flagged",
            "confidence": 0.0,
            "method_note": "No raster features found",
            "geometry": polygon,
            "fused_lines_xy": None,
            "optimizer_result": None
        }

    vertices = np.array(polygon.exterior.coords)[:-1]

    optimizer_result = optimize_vertices(
        vertices=vertices,
        fused_lines_xy=fused_lines_xy,
        target_area=target_area,
        config=config
    )

    if optimizer_result["rolled_back"]:
        status = "flagged"
        confidence = 0.0
        method_note = f"Rollback triggered: {optimizer_result['status_gate']}"
        output_geometry = polygon
    else:
        status = "corrected"
        confidence = calculate_calibrated_confidence(optimizer_result)
        method_note = f"Overlap: {optimizer_result['overlap_before']:.1f}% -> {optimizer_result['overlap_after']:.1f}%"
        output_geometry = optimizer_result["polygon"]

    return {
        "plot_number": plot_number,
        "status": status,
        "confidence": confidence,
        "method_note": method_note,
        "geometry": output_geometry,
        "fused_lines_xy": fused_lines_xy,
        "optimizer_result": optimizer_result
    }
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Polygon

def batch_process_village(input_geojson, tiff_path, output_geojson):
    # 1. Load Data
    print("Loading village dataset...")
    gdf = gpd.read_file(input_geojson)
    
    # Ensure source CRS matches for processing
    with rasterio.open(tiff_path) as src:
        target_crs = src.crs
        if gdf.crs != target_crs:
            print(f"Reprojecting GeoJSON from {gdf.crs} to {target_crs}...")
            gdf = gdf.to_crs(target_crs)
            
        results = []
        
        # 2. Iterate through every plot
        for idx, row in gdf.iterrows():
            print(f"Processing Plot {idx}/{len(gdf)}...")
            
            geom = row.geometry
            # Handle MultiPolygons by picking the largest component
            if geom.geom_type == 'MultiPolygon':
                geom = max(geom.geoms, key=lambda p: p.area)
            
            # Extract target area
            target_area = float(row.get("recorded_area_sqm", geom.area))
            
            # Run optimizer
            # Note: extract_raster_features is the function you already defined
            fused_lines = extract_raster_features(geom, src)
            
            if fused_lines is None:
                # Fallback: No features found
                row['status'] = 'flagged'
                row['confidence'] = 0.0
                results.append(row)
                continue
                
            # Perform optimization
            vertices = np.array(geom.exterior.coords)[:-1]
            res = optimize_vertices(vertices, fused_lines, target_area)
            
            # Update row with results
            row.geometry = res['polygon']
            row['status'] = 'corrected' if not res['rolled_back'] else 'flagged'
            row['confidence'] = calculate_calibrated_confidence(res) if not res['rolled_back'] else 0.0
            row['method_note'] = res['status_gate']
            
            results.append(row)

    # 3. Create output GDF
    out_gdf = gpd.GeoDataFrame(results, crs=target_crs)
    
    # 4. Final CRS conversion to EPSG:4326
    print("Transforming final village dataset to EPSG:4326...")
    out_gdf = out_gdf.to_crs(epsg=4326)
    
    # 5. Save
    out_gdf.to_file(output_geojson, driver='GeoJSON')
    print(f"Processing complete. Saved to {output_geojson}")
import geopandas as gpd

def summarize_results(geojson_path):
    # Load the output file
    gdf = gpd.read_file(geojson_path)
    
    # Count occurrences in the 'status' column
    # Note: Ensure the column name matches what you saved (e.g., 'status')
    status_counts = gdf['status'].value_counts()
    
    total = len(gdf)
    corrected = status_counts.get('corrected', 0)
    flagged = status_counts.get('flagged', 0)
    # If you added the Short-Circuit logic, you might also have this:
    short_circuited = status_counts.get('short-circuited', 0)
    
    print(f"--- Village Optimization Summary ---")
    print(f"Total Features Processed: {total}")
    print(f"Corrected Features:       {corrected}")
    print(f"Flagged Features:         {flagged}")
    if short_circuited > 0:
        print(f"High Fidelity (Skipped):  {short_circuited}")
    print(f"------------------------------------")
    print(f"Success Rate: {((corrected + short_circuited) / total) * 100:.2f}%")

# Execute

# --- Execution ---
# if __name__ == "__main__":
import geopandas as gpd
import matplotlib.pyplot as plt

def plot_comparison(input_path, truth_path, plot_number):
    # 1. Load the files
    input_gdf = gpd.read_file(input_path)
    truth_gdf = gpd.read_file(truth_path)
    pred_gdf = gpd.read_file("optimized_village_4326.geojson")
    print(truth_gdf.head(10))
    # Ensure they have a common identifier
    # If necessary, convert to string to ensure matching
    input_gdf['plot_number'] = input_gdf['plot_number'].astype(str)
    truth_gdf['plot_number'] = truth_gdf['plot_number'].astype(str)
    pred_gdf['plot_number'] = pred_gdf['plot_number'].astype(str)
    # 2. Extract specific plot geometries
    input_plot = input_gdf[input_gdf['plot_number'] == str(plot_number)]
    truth_plot = truth_gdf[truth_gdf['plot_number'] == str(plot_number)]
    pred_plot = pred_gdf[pred_gdf['plot_number'] == str(plot_number)]
    
    if input_plot.empty:
        print(f"Plot number {plot_number} not found in input of the files.")
        return
    elif(truth_plot.empty):
        print(f"Plot number {plot_number} not found in truth  of the files.")
        return


    # 3. Setup the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 4. Plot Input (as a dashed red line)
    input_plot.plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--', label='Input Plot')
    
    # 5. Plot Truth (as a solid green line)
    truth_plot.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=2, label='Truth Plot')
    pred_plot.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2, label='pred Plot')
    
    # 6. Formatting
    plt.title(f"Comparison for Plot #{plot_number}")
    plt.legend()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    
    plt.show()



# --- Execution ---
# Replace '1' with any valid plot_number from your truth file
# plot_comparison("input.geojson", "truth.geojson", plot_number="622")



# -------------------------------THIS IS WHERE I RUN THE CODE FOR THE SCORING--------------------------
# ===================================================================================================================
# import geopandas as gpd
# from types import SimpleNamespace
# from bhume import score 
# pred_gdf = gpd.read_file("optimized_village_4326.geojson")
# truth_gdf = gpd.read_file("maltaradevi.geojson") # Ensure this is your actual truth file
# pred_gdf = pred_gdf.set_index('plot_number')
# truth_gdf = truth_gdf.set_index('plot_number')
# village = SimpleNamespace(
#     slug="Malatavadi",
#     example_truths=truth_gdf, # The 6 ground-truth plots
#     plots=truth_gdf 
# )
# print(score("optimized_village_4326.geojson", village))
# --------------------------------------------------------------------------------------------------------------------------

# PLOTING FOR VISUAL VERIFICATION

# import geopandas as gpd
# import matplotlib.pyplot as plt

# def plot_three_way_comparison(input_path, truth_path, pred_path, plot_number):
#     """
#     Plots Input, Truth, and Prediction for a specific plot_number.
#     """
#     # 1. Load the three files
#     input_gdf = gpd.read_file(input_path)
#     truth_gdf = gpd.read_file(truth_path)
#     pred_gdf = gpd.read_file(pred_path)

#     # 2. Ensure plot_number column is consistent (string format)
#     for gdf in [input_gdf, truth_gdf, pred_gdf]:
#         gdf['plot_number'] = gdf['plot_number'].astype(str)#1145,1403,1476,1710,2647,622

#     # 3. Extract the specific plot
#     input_plot = input_gdf[input_gdf['plot_number'] == str(plot_number)]
#     truth_plot = truth_gdf[truth_gdf['plot_number'] == str(plot_number)]
#     pred_plot = pred_gdf[pred_gdf['plot_number'] == str(plot_number)]

#     if input_plot.empty or truth_plot.empty or pred_plot.empty:
#         print(f"Error: Plot #{plot_number} not found in one or more datasets.")
#         return

#     # 4. Create the plot
#     fig, ax = plt.subplots(figsize=(10, 10))

#     # 5. Plot the three layers
#     # Input (Red Dashed)
#     input_plot.plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--', linewidth=1.5, label='Original Input')
    
#     # Truth (Green Solid - The Gold Standard)
#     truth_plot.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=2.5, label='Ground Truth')
    
#     # Prediction (Blue Solid - Your Result)
#     pred_plot.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2.0, label='Optimized Prediction')

#     # 6. Formatting
#     plt.title(f"Comparison: Plot #{plot_number}\nGreen=Truth, Blue=Optimized, Red=Input")
#     plt.legend()
#     plt.grid(True, linestyle=':', alpha=0.6)
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.show()
# # plot_three_way_comparison("input.geojson","truth.geojson","optimized_village_4326.geojson",2647)#
# # --- Execution ---
# # plot_three_way_comparison("input.geojson", "truth.geojson", "pred2.geojson", plot_number="622")





# # # ==========================================================================================================================
# # 5. EXECUTION & DRY RUN SUITE
# # ==========================================================================================================================
# if __name__ == "__main__":
#     GEOJSON_PATH = "input2.geojson"
#     TIFF_PATH = "new2.tif"
#     PLOT_INDEX = 305
#     config = {
#         "snap_search_radius_meters": 20.0,
#         "edge_sampling_step_meters": 1.5,
#         "max_iterations": 5,
#         "orientation_angle_thresh_deg": 25.0
#     }

#     print(f"Loading datasets for plot visualization...")
#     try:
#         gdf = gpd.read_file(GEOJSON_PATH)
        
#         if PLOT_INDEX >= len(gdf):
#             print(f"Error: PLOT_INDEX {PLOT_INDEX} is out of bounds for dataset of size {len(gdf)}.")
#         else:
#             row = gdf.iloc[PLOT_INDEX]
#             geom_orig = row.geometry

#             if isinstance(geom_orig, MultiPolygon):
#                 geom_orig = max(geom_orig.geoms, key=lambda p: p.area)

#             area_key = "recorded_area_sqm" if "recorded_area_sqm" in row else "area"
#             target_area = float(row[area_key]) if area_key in row else geom_orig.area

#             with rasterio.open(TIFF_PATH) as src:
#                 if gdf.crs != src.crs:
#                     test_gdf = gdf.iloc[[PLOT_INDEX]].to_crs(src.crs)
#                     geom_orig = test_gdf.geometry.values[0]
#                     if isinstance(geom_orig, MultiPolygon):
#                         geom_orig = max(geom_orig.geoms, key=lambda p: p.area)

#                 print(f"Extracting local features for Plot {PLOT_INDEX}...")
#                 fused_lines_xy = extract_raster_features(
#                     polygon=geom_orig, 
#                     raster_src=src, 
#                     image_buffer_meters=60
#                 )

#                 if fused_lines_xy is None:
#                     print("Execution stopped: No raster line features found in proximity.")
#                 else:
#                     print(f"Running edge-centric optimizer loop...")
#                     result = optimize_vertices(
#                         vertices=np.array(geom_orig.exterior.coords)[:-1],
#                         fused_lines_xy=fused_lines_xy,
#                         target_area=target_area,
#                         config=config
#                     )
#                     print(f"--- Plot Debug Stats ---")
#                     print(f"Boundary Overlap: {result['overlap_before']:.2f}% -> {result['overlap_after']:.2f}%")
#                     # Safely handle keys depending on fallback execution
#                     if 'area_deficit_before_pct' in result:
#                         print(f"Area Deficit:     {result['area_deficit_before_pct']:.2f}% -> {result['area_deficit_after_pct']:.2f}%")
#                     print(f"Was Rolled Back?: {result['rolled_back']}")
#                     print(f"Gate Trigger:     {result['status_gate']}")
#                     print(f"Generating diagnostic rays map...")

#                     plot_edge_voting_diagnostics(
#                         original_polygon=geom_orig,
#                         corrected_polygon=result["polygon"],
#                         fused_lines_xy=fused_lines_xy,
#                         viz_data=result["viz_data"],
#                         config=config
#                     )
#     except Exception as e:
#         print(f"Execution Error: {e}")



# -----------------------------------THIS RENDERS THE COMPLETE VILLAGE AND SAVES THE FILE AS GEOJSON---------------------------
if __name__ == "__main__":
    INPUT_GEOJSON = "input.geojson"
    TIFF_PATH = "boundaries.tif"
    OUTPUT_GEOJSON = "optimized_village_4326.geojson"
    
    # Use the same config that worked for your individual plot tests
    config = {
        "snap_search_radius_meters": 25.0,
        "edge_sampling_step_meters": 1.5,
        "max_iterations": 5,
        "orientation_angle_thresh_deg": 25.0,
        "corner_angle_threshold_deg": 110.0,
        "max_vertex_move_meters": 60.0
    }

    print("Starting batch optimization for the entire village...")
    
    # Run the batch processor
    batch_process_village(INPUT_GEOJSON, TIFF_PATH, OUTPUT_GEOJSON)
    
    # Run the summary to see how many were corrected vs flagged
    summarize_results(OUTPUT_GEOJSON)