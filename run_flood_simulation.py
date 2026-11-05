"""
High-Resolution 2D Hydrodynamic Overland Flow Simulation
Location: Kamaraj Nagar, Puducherry, India
Domain: 548 x 491 grid, 5.0m resolution (EPSG:32644)
"""

import os
import time
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
import geopandas as gpd
from scipy.ndimage import maximum_filter
import numba

def run_simulation():
    print("=" * 70)
    print("  KAMARAJ NAGAR 2D HYDRODYNAMIC URBAN FLOOD SIMULATION")
    print("=" * 70)
    
    output_dir = "simulation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Rasters
    print("\n[1/5] Loading geospatial datasets...")
    stamped_path = "KamarajNagar_Flood_Simulator-20261104T120851Z-1-001/KamarajNagar_Flood_Simulator/kamaraj_solver_stamped_dem_5m.tif"
    bare_path = "KamarajNagar_Flood_Simulator-20261104T120851Z-1-001/KamarajNagar_Flood_Simulator/kamaraj_bare_earth_dtm_5m.tif"
    rough_path = "KamarajNagar_Flood_Simulator-20261104T120851Z-1-001/KamarajNagar_Flood_Simulator/kamaraj_hydrology_roughness_5m.tif"
    drain_path = "drain_mask.tif"
    road_path = "road_mask.tif"
    
    with rasterio.open(stamped_path) as s:
        z_dem = s.read(1).astype(np.float64)
        meta = s.meta.copy()
        transform = s.transform
        crs = s.crs
        
    with rasterio.open(bare_path) as s:
        z_dtm = s.read(1).astype(np.float64)
        
    with rasterio.open(rough_path) as s:
        n_rough = s.read(1).astype(np.float64)
        
    with rasterio.open(drain_path) as s:
        drain_arr = s.read(1).astype(np.float64)
        
    with rasterio.open(road_path) as s:
        road_arr = s.read(1).astype(np.float64)
        
    rows, cols = z_dem.shape
    dx = 5.0  # meters
    cell_area = dx * dx  # 25 m^2
    
    # Handle NaNs and boundaries
    valid_mask = np.isfinite(z_dem) & np.isfinite(z_dtm)
    # Fill NaN boundaries with edge values for stable computation
    z_dem_calc = np.nan_to_num(z_dem, nan=999.0)
    z_dtm_calc = np.nan_to_num(z_dtm, nan=999.0)
    n_rough_calc = np.nan_to_num(n_rough, nan=0.035)
    n_rough_calc = np.clip(n_rough_calc, 0.015, 0.10)
    drain_mask_calc = (drain_arr == 1.0) & valid_mask
    road_mask_calc = (road_arr == 1.0) & valid_mask
    
    building_mask = (z_dem_calc - z_dtm_calc > 0.5) & valid_mask
    print(f"  Domain Grid: {rows} rows x {cols} cols (Total cells: {rows*cols:,})")
    print(f"  Valid Cells: {np.sum(valid_mask):,} ({np.sum(valid_mask)/(rows*cols)*100:.1f}%)")
    print(f"  Ground / Streets: {np.sum(valid_mask & ~building_mask):,} cells")
    print(f"  Stamped Buildings: {np.sum(building_mask):,} cells")
    print(f"  Drainage Cells: {np.sum(drain_mask_calc):,} cells")
    
    # 2. Simulation Scenario Configuration
    # 2-hour monsoon cloudburst scenario (Puducherry extreme event)
    # 0 to 45 min: Heavy cloudburst 85 mm/hr
    # 45 to 75 min: Receding rain 30 mm/hr
    # 75 to 120 min: Zero rain (drainage and ponding settlement)
    total_time_sec = 7200  # 2 hours
    dt = 1.0  # 1.0s time step for hydrodynamic stability
    total_steps = int(total_time_sec / dt)
    
    def get_rain_rate(t_sec):
        t_min = t_sec / 60.0
        if t_min <= 45.0:
            # 85 mm/hr peak cloudburst
            return 85.0 / 1000.0 / 3600.0  # m/s
        elif t_min <= 75.0:
            # Linear recession to 20 mm/hr
            frac = (t_min - 45.0) / 30.0
            mmhr = 85.0 - frac * (85.0 - 20.0)
            return mmhr / 1000.0 / 3600.0
        else:
            return 0.0  # rain ceased
            
    # Drain capacity: 30 mm/hr = 30 / 1000 / 3600 m/s
    drain_rate_ms = 30.0 / 1000.0 / 3600.0
    # Soil infiltration rate in vegetated yards: 5 mm/hr
    infil_rate_ms = 5.0 / 1000.0 / 3600.0
    
    print(f"\n[2/5] Initializing Hydrodynamic Solver...")
    print(f"  Total Duration: {total_time_sec/3600:.1f} hours ({total_steps:,} steps at dt={dt}s)")
    print(f"  Storm Profile: 85 mm/hr Peak Cloudburst (45 min) -> Recession (30 min) -> Post-storm Settlement (45 min)")
    print(f"  Drainage Capacity: 30 mm/hr on drainage corridors")
    
    # JIT Compiled 2D Hydrodynamic Kernel
    @numba.njit(fastmath=True)
    def hydrodynamic_step(h, z, n_rough, is_drain, is_ground, is_valid, dt, dx, rain, drain_rate, infil_rate):
        rows, cols = h.shape
        wse = z + h
        
        # Outflow fluxes per cell in 4 directions: East, West, South, North
        flux_E = np.zeros((rows, cols), dtype=np.float64)
        flux_W = np.zeros((rows, cols), dtype=np.float64)
        flux_S = np.zeros((rows, cols), dtype=np.float64)
        flux_N = np.zeros((rows, cols), dtype=np.float64)
        
        # Velocity trackers
        vel_x = np.zeros((rows, cols), dtype=np.float64)
        vel_y = np.zeros((rows, cols), dtype=np.float64)
        
        # 1. Calculate x-direction fluxes (East-West)
        for i in range(rows):
            for j in range(cols - 1):
                if not (is_valid[i, j] and is_valid[i, j+1]):
                    continue
                w1 = wse[i, j]
                w2 = wse[i, j+1]
                diff = w1 - w2
                if abs(diff) < 1e-4:
                    continue
                    
                z_obs = max(z[i, j], z[i, j+1])
                if diff > 0:
                    hf = w1 - z_obs
                    if hf > 1e-3:
                        s = diff / dx
                        n_val = 0.5 * (n_rough[i, j] + n_rough[i, j+1])
                        q = (1.0 / n_val) * (hf ** 1.666667) * np.sqrt(s)
                        flux_E[i, j] = q * dx
                        vel_x[i, j] = max(vel_x[i, j], q / hf)
                else:
                    hf = w2 - z_obs
                    if hf > 1e-3:
                        s = -diff / dx
                        n_val = 0.5 * (n_rough[i, j] + n_rough[i, j+1])
                        q = (1.0 / n_val) * (hf ** 1.666667) * np.sqrt(s)
                        flux_W[i, j+1] = q * dx
                        vel_x[i, j+1] = max(vel_x[i, j+1], q / hf)
                        
        # 2. Calculate y-direction fluxes (South-North)
        for i in range(rows - 1):
            for j in range(cols):
                if not (is_valid[i, j] and is_valid[i+1, j]):
                    continue
                w1 = wse[i, j]
                w2 = wse[i+1, j]
                diff = w1 - w2
                if abs(diff) < 1e-4:
                    continue
                    
                z_obs = max(z[i, j], z[i+1, j])
                if diff > 0:
                    hf = w1 - z_obs
                    if hf > 1e-3:
                        s = diff / dx
                        n_val = 0.5 * (n_rough[i, j] + n_rough[i+1, j])
                        q = (1.0 / n_val) * (hf ** 1.666667) * np.sqrt(s)
                        flux_S[i, j] = q * dx
                        vel_y[i, j] = max(vel_y[i, j], q / hf)
                else:
                    hf = w2 - z_obs
                    if hf > 1e-3:
                        s = -diff / dx
                        n_val = 0.5 * (n_rough[i, j] + n_rough[i+1, j])
                        q = (1.0 / n_val) * (hf ** 1.666667) * np.sqrt(s)
                        flux_N[i+1, j] = q * dx
                        vel_y[i+1, j] = max(vel_y[i+1, j], q / hf)
                        
        # 3. Apply cell-storage flux limiter to guarantee stability and mass conservation
        for i in range(rows):
            for j in range(cols):
                if not is_valid[i, j]:
                    continue
                avail_vol = h[i, j] * (dx * dx)
                tot_out_vol = (flux_E[i, j] + flux_W[i, j] + flux_S[i, j] + flux_N[i, j]) * dt
                if tot_out_vol > 0.0 and tot_out_vol > avail_vol * 0.95:
                    scale = (avail_vol * 0.95) / tot_out_vol
                    flux_E[i, j] *= scale
                    flux_W[i, j] *= scale
                    flux_S[i, j] *= scale
                    flux_N[i, j] *= scale
                    
        # 4. Update water depths & mass conservation
        h_new = np.copy(h)
        for i in range(rows):
            for j in range(cols):
                if not is_valid[i, j]:
                    continue
                    
                # Inflows
                inflow_vol = 0.0
                if j > 0:
                    inflow_vol += flux_E[i, j-1] * dt
                if j < cols - 1:
                    inflow_vol += flux_W[i, j+1] * dt
                if i > 0:
                    inflow_vol += flux_S[i-1, j] * dt
                if i < rows - 1:
                    inflow_vol += flux_N[i+1, j] * dt
                    
                # Outflows
                outflow_vol = (flux_E[i, j] + flux_W[i, j] + flux_S[i, j] + flux_N[i, j]) * dt
                
                # Net advection
                dh = (inflow_vol - outflow_vol) / (dx * dx)
                
                # Source (Rain) and Sinks (Drainage & Infiltration)
                source_sink = rain * dt
                if is_drain[i, j] and h[i, j] > 1e-4:
                    sink = min(h[i, j], drain_rate * dt)
                    source_sink -= sink
                elif is_ground[i, j] and n_rough[i, j] >= 0.045 and h[i, j] > 1e-4:
                    sink = min(h[i, j], infil_rate * dt)
                    source_sink -= sink
                    
                h_new[i, j] = max(0.0, h[i, j] + dh + source_sink)
                
        # Total cell velocity magnitude
        vel = np.sqrt(vel_x**2 + vel_y**2)
        return h_new, vel

    # State variables
    h = np.zeros((rows, cols), dtype=np.float64)
    h_max = np.zeros((rows, cols), dtype=np.float64)
    v_max = np.zeros((rows, cols), dtype=np.float64)
    
    # Snapshot intervals (in minutes)
    snapshots = [10, 20, 30, 45, 60, 75, 90, 105, 120]
    snapshot_steps = {int(m * 60 / dt): m for m in snapshots}
    saved_frames = {}
    
    print("\n[3/5] Running Dynamic 2D Overland Flow Simulation...")
    start_sim_time = time.time()
    
    # Warmup Numba compilation
    _h, _v = hydrodynamic_step(h, z_dem_calc, n_rough_calc, drain_mask_calc, ~building_mask, valid_mask, dt, dx, 0.0, drain_rate_ms, infil_rate_ms)
    print("  JIT Compilation complete. Executing timesteps...")
    
    report_interval = int(600 / dt)  # every 10 min
    
    for step in range(1, total_steps + 1):
        sim_sec = step * dt
        r_rate = get_rain_rate(sim_sec)
        
        h, v = hydrodynamic_step(
            h, z_dem_calc, n_rough_calc, drain_mask_calc, 
            ~building_mask, valid_mask, dt, dx, r_rate, 
            drain_rate_ms, infil_rate_ms
        )
        
        # Track maximum flood depth on ground/streets
        # Roofs have transient rain that sheds; true flood hazard is ground water depth
        ground_h = np.where(~building_mask & valid_mask, h, 0.0)
        h_max = np.maximum(h_max, ground_h)
        v_max = np.maximum(v_max, np.where(~building_mask & valid_mask, v, 0.0))
        
        if step in snapshot_steps:
            m = snapshot_steps[step]
            saved_frames[m] = ground_h.astype(np.float32)
            flooded_area_m2 = np.sum(ground_h > 0.05) * cell_area
            print(f"  Step {step:5d} / {total_steps} (T = {m:3d} min) | Rain: {r_rate*1000*3600:4.1f} mm/hr | Max Depth: {np.nanmax(ground_h):.2f}m | Flooded Area: {flooded_area_m2/10000:.1f} ha")
            
    sim_duration = time.time() - start_sim_time
    print(f"  Simulation completed in {sim_duration:.2f} seconds ({total_steps/sim_duration:.1f} steps/sec)!")
    
    # 4. Export GeoTIFFs
    print("\n[4/5] Exporting Simulation Rasters (GeoTIFF)...")
    out_meta = meta.copy()
    out_meta.update({
        'dtype': 'float32',
        'nodata': -9999.0,
        'compress': 'lzw'
    })
    
    h_max_export = np.where(valid_mask, h_max, -9999.0).astype(np.float32)
    v_max_export = np.where(valid_mask, v_max, -9999.0).astype(np.float32)
    
    depth_path = os.path.join(output_dir, "flood_depth_max.tif")
    with rasterio.open(depth_path, "w", **out_meta) as dst:
        dst.write(h_max_export, 1)
    print(f"  Saved: {depth_path}")
    
    vel_path = os.path.join(output_dir, "flood_velocity_max.tif")
    with rasterio.open(vel_path, "w", **out_meta) as dst:
        dst.write(v_max_export, 1)
    print(f"  Saved: {vel_path}")
    
    # Save key time frames as rasters
    for m in [30, 60, 90, 120]:
        t_path = os.path.join(output_dir, f"flood_depth_t{m:03d}m.tif")
        t_arr = np.where(valid_mask, saved_frames[m], -9999.0).astype(np.float32)
        with rasterio.open(t_path, "w", **out_meta) as dst:
            dst.write(t_arr, 1)
        print(f"  Saved: {t_path}")
        
    # 5. Building Inundation Analysis
    print("\n[5/5] Conducting Building Impact Assessment (26,236 buildings)...")
    bldg_gdf = gpd.read_file("kamaraj_buildings_clean.geojson")
    bldg_gdf_proj = bldg_gdf.to_crs(crs)
    
    # Calculate perimeter / contact flood depth using 3x3 neighborhood maximum of ground flood depth
    h_max_dilated = maximum_filter(h_max, size=3)
    
    # Sample at building centroids
    inv_tf = ~transform
    centroids = bldg_gdf_proj.geometry.centroid
    cols_samp, rows_samp = inv_tf * (centroids.x, centroids.y)
    cols_samp = np.clip(np.round(cols_samp).astype(int), 0, cols - 1)
    rows_samp = np.clip(np.round(rows_samp).astype(int), 0, rows - 1)
    
    sampled_depths = h_max_dilated[rows_samp, cols_samp]
    # Update max_flood_ in meters (rounded to 3 decimal places)
    bldg_gdf["max_flood_"] = np.round(sampled_depths, 3)
    
    # Categorize Risk
    conditions = [
        (bldg_gdf["max_flood_"] < 0.05),
        (bldg_gdf["max_flood_"] >= 0.05) & (bldg_gdf["max_flood_"] < 0.30),
        (bldg_gdf["max_flood_"] >= 0.30) & (bldg_gdf["max_flood_"] < 0.60),
        (bldg_gdf["max_flood_"] >= 0.60) & (bldg_gdf["max_flood_"] < 1.20),
        (bldg_gdf["max_flood_"] >= 1.20)
    ]
    choices = ["Safe / Dry", "Minor (0.05-0.3m)", "Moderate (0.3-0.6m)", "High (0.6-1.2m)", "Severe (>1.2m)"]
    bldg_gdf["flood_risk"] = np.select(conditions, choices, default="Safe / Dry")
    
    # Save updated GeoJSON
    flooded_bldgs_path = os.path.join(output_dir, "kamaraj_buildings_flooded.geojson")
    bldg_gdf.to_file(flooded_bldgs_path, driver="GeoJSON")
    print(f"  Saved Updated Buildings GeoJSON: {flooded_bldgs_path}")
    
    # Risk summary statistics
    risk_counts = bldg_gdf["flood_risk"].value_counts().to_dict()
    total_bldgs = len(bldg_gdf)
    flooded_bldgs = total_bldgs - risk_counts.get("Safe / Dry", 0)
    
    flooded_area_gt_05 = np.sum((h_max > 0.05) & valid_mask) * cell_area
    flooded_area_gt_30 = np.sum((h_max > 0.30) & valid_mask) * cell_area
    flooded_area_gt_60 = np.sum((h_max > 0.60) & valid_mask) * cell_area
    
    stats = {
        "simulation_scenario": "Puducherry Northeast Monsoon Severe Cloudburst",
        "duration_minutes": total_time_sec / 60,
        "peak_rain_rate_mm_hr": 85.0,
        "drain_capacity_mm_hr": 30.0,
        "grid_resolution_m": dx,
        "domain_cells": int(rows * cols),
        "valid_cells": int(np.sum(valid_mask)),
        "max_depth_recorded_m": float(np.nanmax(h_max)),
        "max_velocity_recorded_m_s": float(np.nanmax(v_max)),
        "inundation_areas": {
            "depth_gt_0.05m_ha": round(flooded_area_gt_05 / 10000.0, 2),
            "depth_gt_0.30m_ha": round(flooded_area_gt_30 / 10000.0, 2),
            "depth_gt_0.60m_ha": round(flooded_area_gt_60 / 10000.0, 2),
        },
        "building_impact": {
            "total_buildings": total_bldgs,
            "flooded_buildings_total": int(flooded_bldgs),
            "flooded_percentage": round(flooded_bldgs / total_bldgs * 100.0, 1),
            "risk_breakdown": risk_counts
        },
        "time_series_snapshots": snapshots
    }
    
    stats_path = os.path.join(output_dir, "simulation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved Statistics: {stats_path}")
    
    # Save downsampled/quantized time series grids for the WebMap
    print("\n  Preparing WebMap animation frames...")
    frames_dict = {}
    for m in snapshots:
        frame_grid = saved_frames[m]
        # Quantize to 2 decimal places to minimize JSON size
        # We can store non-zero cells in a sparse or compressed format
        sparse_cells = []
        # Downsample 2x for ultra-smooth 60fps web rendering (274 x 245)
        ds_grid = frame_grid[::2, ::2]
        r_indices, c_indices = np.where(ds_grid >= 0.05)
        for r_i, c_i in zip(r_indices, c_indices):
            val = float(np.round(ds_grid[r_i, c_i], 2))
            sparse_cells.append([int(r_i), int(c_i), val])
        frames_dict[str(m)] = {
            "time_min": m,
            "flooded_cells_count": len(sparse_cells),
            "cells": sparse_cells
        }
        
    web_anim_path = os.path.join(output_dir, "web_flood_frames.json")
    # Downsampled metadata for leaflet overlay bounds
    # Top-left and bottom-right in WGS84
    with rasterio.open(stamped_path) as s:
        bounds_utm = s.bounds
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(bounds_utm.left, bounds_utm.bottom)
    max_lon, max_lat = transformer.transform(bounds_utm.right, bounds_utm.top)
    
    web_meta = {
        "bounds_wgs84": [[min_lat, min_lon], [max_lat, max_lon]],
        "grid_shape": [int(np.ceil(rows/2)), int(np.ceil(cols/2))],
        "stats": stats,
        "frames": frames_dict
    }
    with open(web_anim_path, "w") as f:
        json.dump(web_meta, f)
    print(f"  Saved Web Animation Frames: {web_anim_path}")
    
    print("\n" + "=" * 70)
    print("  SIMULATION & ANALYSIS COMPLETE!")
    print(f"  Total Buildings Flooded: {flooded_bldgs:,} / {total_bldgs:,} ({flooded_bldgs/total_bldgs*100:.1f}%)")
    print(f"  Peak Flood Depth: {np.nanmax(h_max):.2f} meters")
    print(f"  Total Flooded Area (>0.05m): {flooded_area_gt_05/10000:.1f} hectares")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation()

