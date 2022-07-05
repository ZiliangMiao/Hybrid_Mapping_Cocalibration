#include <string>
#include <vector>
#include <pcl/common/common.h>
/** ros **/
#include <ros/ros.h>
#include <ros/package.h>
using namespace std;
typedef pcl::PointXYZRGB RGBPointT;
typedef pcl::PointCloud<RGBPointT> RGBCloudT;
typedef pcl::PointCloud<RGBPointT>::Ptr RGBCloudPtr;

class FisheyeProcess{
public:
    /** essential params **/
    const string kPkgPath = ros::package::getPath("calibration");
    const string kDatasetPath = this->kPkgPath + "/data/floor5";
    int spot_idx = 0;
    int view_idx = 0;
    int num_spots = 4;
    int num_views = 3;
    int fullview_idx = (this->num_views-1) / 2;
    vector<vector<string>> pose_folder_path_vec;

    /** original data - images **/
    const int kFisheyeRows = 2048;
    const int kFisheyeCols = 2448;
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
        int label; /** label = 0 -> empty pixel; label = 1 -> normal pixel **/
        int num_pts; /** number of points **/
        vector<int> pts_indices;
    }Tags; /** "Tags" here is a struct type, equals to "struct Tags", LidarProcess::Tags **/
    typedef vector<vector<Tags>> TagsMap;
    vector<vector<TagsMap>> tags_map_vec; /** container of tagsMaps of each pose **/

    /***** Intrinsic Parameters *****/
    struct Intrinsic {
        double a1 = 0;
        double a0 = 6.073762e+02;
        double a2 = -5.487830e-04;
        double a3 = -2.809080e-09;
        double a4 = -1.175734e-10;
        double c = 1.000143;
        double d = -0.000177;
        double e = 0.000129;
        double u0 = 1022.973079;
        double v0 = 1200.975472;
    } intrinsic;

    /********* File Path of the Specific Pose *********/
    struct PoseFilePath {
        PoseFilePath () = default;
        PoseFilePath (const string &pose_folder_path) {
            this->output_folder_path = pose_folder_path + "/outputs";
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
    cv::Mat ReadFisheyeImage();
    std::tuple<RGBCloudPtr, RGBCloudPtr> FisheyeImageToSphere();
    std::tuple<RGBCloudPtr, RGBCloudPtr> FisheyeImageToSphere(cv::Mat &image);
    void SphereToPlane(RGBCloudPtr sphere_polar_cloud);
    void SphereToPlane(RGBCloudPtr sphere_polar_cloud, double bandwidth);

    /** Edge Related **/
    void ReadEdge();
    void EdgeToPixel();
    void PixLookUp(RGBCloudPtr fisheye_pixel_cloud);
    std::vector<double> Kde(double bandwidth, double scale, bool polar);
    int EdgeExtraction(string pkg_path, int mode);

    /** Get and Set Methods **/
    void SetIntrinsic(vector<double> parameters) {
        /** polynomial params **/
        this->intrinsic.a0 = parameters[6];
        this->intrinsic.a2 = parameters[7];
        this->intrinsic.a3 = parameters[8];
        this->intrinsic.a4 = parameters[9];
        /** expansion and distortion **/
        this->intrinsic.c = parameters[10];
        this->intrinsic.d = parameters[11];
        this->intrinsic.e = parameters[12];
        /** center **/
        this->intrinsic.u0 = parameters[13];
        this->intrinsic.v0 = parameters[14];
    }

    void SetSpotIdx(int spot_idx) {
        this->spot_idx = spot_idx;
    }

    void SetViewIdx(int view_idx) {
        this->view_idx = view_idx;
    }
};
