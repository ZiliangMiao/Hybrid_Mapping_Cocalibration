import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target])


print("Load two point clouds and show initial pose ...")
# source = o3d.io.read_point_cloud("/home/godm/catkin_ws/src/lidar_fisheye_fusion_github/calibration/data/lh3_global/spot1/fullview_recon/spot_rgb_cloud.pcd")
# target = o3d.io.read_point_cloud("/home/godm/catkin_ws/src/lidar_fisheye_fusion_github/calibration/data/lh3_global/spot0/fullview_recon/spot_rgb_cloud.pcd")
source = o3d.io.read_point_cloud("/home/godm/catkin_ws/src/lidar_fisheye_fusion_github/calibration/data/lh3_global/fullview_dense_rgb_clouds/spot_rgb_cloud_spot1.pcd")
target = o3d.io.read_point_cloud("/home/godm/catkin_ws/src/lidar_fisheye_fusion_github/calibration/data/lh3_global/fullview_dense_rgb_clouds/spot_rgb_cloud_spot0.pcd")


if __name__ == "__main__":
    # Draw initial alignment.
    current_transformation = np.loadtxt("/home/godm/catkin_ws/src/lidar_fisheye_fusion_github/calibration/data/lh3_global/lio_mid360/lio_spot_trans_mat_1_0.txt")
    print("1. Downsample with a voxel size %.2f" % 0.05)
    source_down = source.voxel_down_sample(0.05)
    target_down = target.voxel_down_sample(0.05)
    draw_registration_result(source, target, current_transformation)

    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [100, 70, 40]
    print("Colored point cloud registration ...\n")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("2. Estimate normal")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp, "\n")
    draw_registration_result(source_down, target_down, result_icp.transformation)
    print(current_transformation)
