#ifndef _LIDAREDGE_H
#define _LIDAREDGE_H
#include <fstream>
#include <iostream>
#include <sstream>
#include <std_msgs/Header.h>
#include <string>
#include <vector>

using namespace std;

class LidarProcess{
public:
    string topic_name = "/livox/lidar";
    /***** Variables of LidarProcess Class *****/
    /** tags and maps **/
    typedef struct Tags
    {
        int label; /** label = 0 -> empty pixel; label = 1 -> normal pixel **/
        int num_pts; /** number of points **/
        vector<int> pts_indices;
        double mean;
        double sigma; /** sigma is the standard deviation estimation of lidar edge distribution **/
        double weight;
        int num_hidden_pts;
    }Tags; /** "Tags" here is a struct type, equals to "struct Tags", LidarProcess::Tags **/
    typedef vector<vector<Tags>> TagsMap;
    vector<TagsMap> tags_map_vec; /** container of tagsMaps of each scene **/

    /** const parameters - original data - images and point clouds **/
    const bool byIntensity = true;
    static const int numPcds = 500;
    const int flatRows = int((double)110/90 * 1000) + 1;
    const int flatCols = 4000;
    const double radPerPix = (M_PI/2) / 1000;

    /** coordinates of edge pixels (which are considered as the edge) **/
    typedef vector<vector<int>> EdgePixels;
    vector<EdgePixels> edge_pixels_vec;

    /** spatial coordinates of edge points (center of distribution) **/
    typedef vector<vector<double>> EdgePts;
    vector<EdgePts> edge_pts_vec;

    /** mean position of the lidar pts in a specific pixel space **/
    typedef pcl::PointCloud<pcl::PointXYZI>::Ptr EdgeCloud; /** note: I is used to store the weight **/
    vector<EdgeCloud> edge_cloud_vec; /** container of edgeClouds of each scene **/

    /***** Extrinsic Parameters *****/
    struct Extrinsic {
        double rx = 0;
        double ry = 0;
        double rz = 0;
        double tx = 0;
        double ty = 0;
        double tz = 0;
    } extrinsic;

    /***** Data of Multiple Scenes *****/
    int scene_idx = 0;
    int num_scenes = 5;
    vector<string> scenes_path_vec;

    /** File Path of the Specific Scene **/
    struct SceneFilePath
    {
        SceneFilePath(string ScenePath) {
            this -> output_folder_path = ScenePath + "/outputs";
            this -> pcds_folder_path = ScenePath + "/pcds";
            this -> edge_img_path = ScenePath + "/edges/lidEdge.png";
            this -> result_folder_path = ScenePath + "/results";
            this -> proj_folder_path = this -> output_folder_path + "/byIntensity";
            this -> dense_pcd_path = this -> output_folder_path + "/lidDense" + to_string(numPcds) + ".pcd";
            this -> flat_img_path = this -> proj_folder_path + "/flatLidarImage.bmp";
            this -> polar_pcd_path = this -> proj_folder_path + "/lidPolar.pcd";
            this -> cart_pcd_path = this -> proj_folder_path + "/lidCartesian.pcd";
            this -> tags_map_path = this -> proj_folder_path + "/tags_map.txt";
            this -> edge_flat_pixels_path = this -> output_folder_path + "/lidEdgePix.txt";
            this -> edge_points_coordinates_path = this -> output_folder_path + +"/lid3dOut.txt";
            this -> edge_fisheye_projection_path = this -> output_folder_path + "/lidTrans.txt";
            this -> params_record_path = this -> output_folder_path + "/ParamsRecord.txt";
        }
        string output_folder_path;
        string pcds_folder_path;
        string edge_img_path;
        string result_folder_path;
        string proj_folder_path;
        string dense_pcd_path;
        string flat_img_path;
        string polar_pcd_path;
        string cart_pcd_path;
        string tags_map_path;
        string edge_flat_pixels_path;
        string edge_points_coordinates_path;
        string edge_fisheye_projection_path;
        string params_record_path;
    };
    vector<struct SceneFilePath> scenes_files_path_vec;

public:
    LidarProcess(string dataPath, const bool byIntensity);
    /***** Point Cloud Generation *****/
    int ReadFileList(const std::string &folder_path, std::vector<std::string> &file_list);
    void CreateDensePcd();
    void BagToPcd(string bag_file);

    /***** Edge Related *****/
    void EdgeToPixel();
    void ReadEdge();
    vector<vector<double>> EdgeCloudProjectToFisheye(vector<double> _p);
    vector<double> Kde(vector<vector<double>> edge_pixels, int row_samples, int col_samples);

    /***** LiDAR Pre-Processing *****/
    std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> LidarToSphere();
    void SphereToPlaneRNN(pcl::PointCloud<pcl::PointXYZI>::Ptr polar_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr cart_cloud);
    void PixLookUp(pcl::PointCloud<pcl::PointXYZI>::Ptr cart_cloud);

    /***** Get and Set Methods *****/
    void SetExtrinsic(vector<double> _p) {
        this -> extrinsic.rx = _p[0];
        this -> extrinsic.ry = _p[1];
        this -> extrinsic.rz = _p[2];
        this -> extrinsic.tx = _p[3];
        this -> extrinsic.ty = _p[4];
        this -> extrinsic.tz = _p[5];
    }

    void SetSceneIdx(int scene_idx) {
        this -> scene_idx = scene_idx;
    }
};
#endif