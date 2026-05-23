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
    # Server 1: Curvature ON, surface-aware OFF (sigma_theta=0)
    # Tests whether surface-aware kernel is the problem
    common_args = "-r2 --ncc_scale 0.5 --use_gms --gms_position_mode free --gms_appearance_mode group --gms_normal_mode rotation --gms_lambda_align 5e-2 --gms_lambda_surf 1e-2 --gms_lambda_curv 1e-4 --gms_top_m 4 --gms_sigma_theta 0"
    # out_name = "gms_curv_no_surfaware"
    
    # Server 2: Curvature ON, lambda_surf=0
    # Tests whether surface position loss is hurting
    common_args = "-r2 --ncc_scale 0.5 --use_gms --gms_position_mode free --gms_appearance_mode group --gms_normal_mode rotation --gms_lambda_align 5e-2 --gms_lambda_surf 0 --gms_lambda_curv 1e-4 --gms_top_m 4 --gms_sigma_theta 0.3"
    # out_name = "gms_curv_no_surfloss"
        
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