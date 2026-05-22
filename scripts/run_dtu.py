import os

# scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
scenes = [65]
data_base_path='/workspace/colmap_scenes/DTU/dtu'
out_base_path='/workspace/colmap_scenes/DTU/dtu_output'
eval_path='/workspace/colmap_scenes/DTU/dtu_eval'
out_name='gms'
gpu_id=0

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/dtu_scan{scene}/{out_name}/*'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet -r2 --ncc_scale 0.5"
    # common_args = "-r2 --ncc_scale 0.5 --use_gms"
    
    # Server 1: scan65 with RESCALED anisotropic, C2-equivalent settings
    # Tests whether the rescale fix restored baseline
    # common_args = "-r2 --ncc_scale 0.5 --use_gms --gms_position_mode free --gms_appearance_mode group --gms_normal_mode rotation --gms_lambda_align 5e-2 --gms_lambda_plane 0 --gms_top_m 4"
    # scene = 65, out_name = "gms_aniso_rescaled"
    # Expected: ~0.537 (restored) or better. If >0.55, anisotropic is still broken.
    
    # Server 2: scan24 with RESCALED anisotropic, C2 settings
    # Tests whether C2 helps or hurts on the easy scene
    # common_args = "-r2 --ncc_scale 0.5 --use_gms --gms_position_mode free --gms_appearance_mode group --gms_normal_mode rotation --gms_lambda_align 5e-2 --gms_lambda_plane 0 --gms_top_m 4"
    # scene = 24, out_name = "gms_aniso_C2"
    # Expected: 0.33-0.36. If <0.33, we beat PGSR on easy too.
    
    # Server 3: scan83 (plant, PGSR=1.08) with C2
    # Tests our method on the hardest DTU scene to see if hard scenes benefit most
    # common_args = "-r2 --ncc_scale 0.5 --use_gms --gms_position_mode free --gms_appearance_mode group --gms_normal_mode rotation --gms_lambda_align 5e-2 --gms_lambda_plane 0 --gms_top_m 4"
    # scene = 83, out_name = "gms_aniso_C2"
    # Expected: 0.95-1.10. Big absolute headroom means meaningful absolute improvement possible.
    
    # Server 4: scan65 with sigma_n=0.025 (tighter perpendicular bandwidth)
    # Currently untested; might help if groups should be more committed perpendicular-wise
    # common_args = "-r2 --ncc_scale 0.5 --use_gms --gms_position_mode free --gms_appearance_mode group --gms_normal_mode rotation --gms_lambda_align 5e-2 --gms_lambda_plane 0 --gms_top_m 4 --gms_sigma_n 0.025"
    # scene = 65, out_name = "gms_sigma0025"
    # Expected: 0.50-0.56. Could go either way.
    
    # Server 5: scan65 with lambda_align=1e-1 + rescaled anisotropic
    # We tested 1e-1 with old isotropic (=0.550). Worth testing with new rho.
    # common_args = "-r2 --ncc_scale 0.5 --use_gms --gms_position_mode free --gms_appearance_mode group --gms_normal_mode rotation --gms_lambda_align 1e-1 --gms_lambda_plane 0 --gms_top_m 4"
    # scene = 65, out_name = "gms_align_1e-1_aniso"
    # Expected: 0.51-0.56. Tests whether stronger align + better rho compound.
        
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/dtu_scan{scene}/{out_name} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/dtu_scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/dtu_scan{scene}/{out_name}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}" #+ " --use_icp"
    print(cmd)
    os.system(cmd)