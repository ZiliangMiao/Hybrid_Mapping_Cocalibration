#include <string>
#include <vector>
#include <pcl/common/common.h>
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZI>::Ptr IntensityCloudPtr; /** note: I is used to store the weight **/
class LidarProcess{
public:
    string topic_name = "/livox/lidar";
    /** tags and maps **/
    typedef struct Tags {
        int label; /** label = 0 -> empty pixel; label = 1 -> normal pixel **/
        int num_pts; /** number of points **/
        vector<int> pts_indices;
        float mean;
        float sigma; /** sigma is the standard deviation estimation of lidar edge distribution **/
        float weight;
        int num_hidden_pts;
    }Tags; /** "Tags" here is a struct type, equals to "struct Tags", LidarProcess::Tags **/
    typedef vector<vector<Tags>> TagsMap;
    vector<TagsMap> tags_map_vec; /** container of tagsMaps of each scene **/

    /** const parameters - original data - images and point clouds **/
    const bool kProjByIntensity = true;
    static const int kNumPcds = 14;
    const int kFlatRows = 2000;
    const int kFlatCols = 4000;
    const double kRadPerPix = (M_PI / 2) / 1000;

    /** coordinates of edge pixels (which are considered as the edge) **/
    typedef vector<vector<int>> EdgePixels;
    vector<EdgePixels> edge_pixels_vec;

    /** spatial coordinates of edge points (center of distribution) **/
    typedef vector<vector<double>> EdgePts;
    vector<EdgePts> edge_pts_vec;

    /** mean position of the lidar pts in a specific pixel space **/
    vector<IntensityCloudPtr> edge_cloud_vec; /** container of edgeClouds of each scene **/

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
    int num_scenes = 7;
    vector<string> scenes_path_vec;

    /** File Path of the Specific Scene **/
    struct SceneFilePath {
        SceneFilePath(const string& ScenePath) {
            this -> output_folder_path = ScenePath + "/outputs";
            this -> pcds_folder_path = ScenePath + "/pcds";
            this -> edge_img_path = ScenePath + "/edges/lidEdge.png";
            this -> result_folder_path = ScenePath + "/results";
            this -> proj_folder_path = this -> output_folder_path + "/byIntensity";
            this -> dense_pcd_path = this -> output_folder_path + "/lidDense" + to_string(kNumPcds) + ".pcd";
            this -> flat_img_path = this -> proj_folder_path + "/flatLidarImage.bmp";
            this -> polar_pcd_path = this -> proj_folder_path + "/lidPolar.pcd";
            this -> cart_pcd_path = this -> proj_folder_path + "/lidCartesian.pcd";
            this -> tags_map_path = this -> proj_folder_path + "/tags_map.txt";
            this -> edge_pts_coordinates_path = this -> output_folder_path + +"/lid3dOut.txt";
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
        string edge_pts_coordinates_path;
        string edge_fisheye_projection_path;
        string params_record_path;
    };
    vector<struct SceneFilePath> scenes_files_path_vec;

    /** Degree Map **/
    std::map<int, int> degree_map;

public:
    LidarProcess(const string& pkg_path);
    /***** Point Cloud Generation *****/
    static int ReadFileList(const string &folder_path, vector<string> &file_list);
    void CreateDensePcd();
    void CreateDensePcd(string full_view_pcd_path);
    void BagToPcd(string bag_file);

    /***** Edge Related *****/
    void EdgeToPixel();
    void ReadEdge();
    vector<vector<double>> EdgeCloudProjectToFisheye(vector<double> _p);
    vector<double> Kde(vector<vector<double>> edge_pixels, int row_samples, int col_samples);

    /***** LiDAR Pre-Processing *****/
    std::tuple<IntensityCloudPtr, IntensityCloudPtr> LidarToSphere();
    void SphereToPlane(const IntensityCloudPtr& polar_cloud, const IntensityCloudPtr& cart_cloud);
    void PixLookUp(const IntensityCloudPtr& cart_cloud);

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