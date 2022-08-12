#include <string>
#include <vector>
#include <thread>
#include <pcl/common/common.h>
/** ros **/
#include <ros/ros.h>
#include <ros/package.h>
#include <spline.h>

using namespace std;
using namespace tk;

/** typedef **/
typedef pcl::PointXYZI PointI;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointI> CloudI;
typedef pcl::PointCloud<PointRGB> CloudRGB;

class FisheyeProcess{
public:
    /** essential params **/
    string kPkgPath = ros::package::getPath("calibration");
    string dataset_name;
    string kDatasetPath;
    int spot_idx = 0;
    int view_idx = 0;
    int num_spots;
    int num_views;
    int view_angle_init;
    int view_angle_step;
    int fullview_idx;
    vector<vector<string>> poses_folder_path_vec;

    /** original data - images **/
    int kFisheyeRows = 2048;
    int kFisheyeCols = 2448;
    const int kFlatRows = int((double)110 / 90 * 1000) + 1;
    const int kFlatCols = 4000;
    const float kRadPerPix = (M_PI * 2) / 4000;

    /** coordinates of edge pixels in flat images **/
    typedef vector<vector<int>> EdgePixels;
    vector<vector<EdgePixels>> edge_pixels_vec;

    /** coordinates of edge pixels in fisheye images **/
    typedef vector<vector<double>> EdgeFisheyePixels;
    vector<vector<EdgeFisheyePixels>> edge_fisheye_pixels_vec;

    /** tagsmap container **/
    typedef struct Tags {
        vector<int> pts_indices = {};
    }Tags; /** "Tags" here is a struct type, equals to "struct Tags", LidarProcess::Tags **/
    typedef vector<vector<Tags>> TagsMap;
    vector<vector<TagsMap>> tags_map_vec; /** container of tagsMaps of each pose **/

    /***** Intrinsic Parameters *****/
    struct Intrinsic {
        double a0 = 0;
        double a1 = 0;
        double a2 = 0;
        double a3 = 0;
        double a4 = 0;
        double c = 1;
        double d = 0;
        double e = 0;
        double u0 = 1024;
        double v0 = 1224;
    } intrinsic;

    /********* File Path of the Specific Pose *********/
    struct PoseFilePath {
        PoseFilePath () = default;
        PoseFilePath (const string &pose_folder_path) {
            this->output_folder_path = pose_folder_path + "/outputs/fisheye_outputs";
            this->fusion_result_folder_path = pose_folder_path + "/results";
            this->fisheye_hdr_img_path = pose_folder_path + "/images/grab_0.bmp";
            this->edge_img_path = pose_folder_path + "/edges/camEdge.png";
            this->flat_img_path = this -> output_folder_path + "/flatImage.bmp";
            this->edge_fisheye_pixels_path = this -> output_folder_path + "/camPixOut.txt";
            this->kde_samples_path = this -> output_folder_path + "/camKDE.txt";
            this->fusion_img_path = this -> fusion_result_folder_path + "/fusion.bmp";
        }
        string output_folder_path;
        string fusion_result_folder_path;
        string fusion_img_path;
        string fisheye_hdr_img_path;
        string edge_img_path;
        string flat_img_path;
        string edge_fisheye_pixels_path;
        string kde_samples_path;
    };
    vector<vector<struct PoseFilePath>> poses_files_path_vec;

    /** Degree Map **/
    std::map<int, int> degree_map;

public:
    FisheyeProcess();
    /** Fisheye Pre-Processing **/
    cv::Mat ReadFisheyeImage(string fisheye_hdr_img_path);
    std::tuple<CloudRGB::Ptr, CloudRGB::Ptr> FisheyeImageToSphere();
    std::tuple<CloudRGB::Ptr, CloudRGB::Ptr> FisheyeImageToSphere(cv::Mat &image, tk::spline spline);
    void SphereToPlane(CloudRGB::Ptr fisheye_polar_cloud);
    void SphereToPlane(CloudRGB::Ptr fisheye_polar_cloud, double bandwidth);

    /** Edge Related **/
    void ReadEdge();
    void EdgeToPixel();
    void PixLookUp(CloudRGB::Ptr fisheye_pixel_cloud);
    std::vector<double> Kde(double bandwidth, double scale);
    void EdgeExtraction();

    /** Get and Set Methods **/
    void SetIntrinsic(vector<double> &parameters) {
        /** center **/
        this->intrinsic.u0 = parameters[6];
        this->intrinsic.v0 = parameters[7];
        /** polynomial params **/
        this->intrinsic.a0 = parameters[8];
        this->intrinsic.a1 = parameters[9];
        this->intrinsic.a2 = parameters[10];
        this->intrinsic.a3 = parameters[11];
        this->intrinsic.a4 = parameters[12];
        /** expansion and distortion **/
        this->intrinsic.c = parameters[13];
        this->intrinsic.d = parameters[14];
        this->intrinsic.e = parameters[15];
    }

    void SetSpotIdx(int spot_idx) {
        this->spot_idx = spot_idx;
    }

    void SetViewIdx(int view_idx) {
        this->view_idx = view_idx;
    }
};
