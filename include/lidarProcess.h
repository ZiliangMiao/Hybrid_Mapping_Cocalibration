#ifndef _LIDAREDGE_H
#define _LIDAREDGE_H
#include <Eigen/Core>
#include <pcl/kdtree/kdtree_flann.h>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h> 
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sstream>
#include <std_msgs/Header.h>
#include <string>
#include <vector>
#include <time.h>
#include <armadillo>

using namespace std;

class lidarProcess{
    public:
        /** topic name **/
        string topicName = "/livox/lidar";

        /***** Variables of lidarProcess Class *****/
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
        }Tags; /** "Tags" here is a struct type, equals to "struct Tags", lidarProcess::Tags **/
        typedef vector<vector<Tags>> TagsMap;
        TagsMap tags_map;
        vector<TagsMap> tags_map_vec; /** container of tagsMaps of each scene **/

        /** original data - images and point clouds **/
        bool byIntensity = true;
        static const int numPcds = 500;
        int flatRows = int((double)110/90 * 1000) + 1;
        int flatCols = 4000;
        double radPerPix = (M_PI/2) / 1000;
        /** coordinates of edge pixels (which are considered as the edge) **/
        typedef vector<vector<int>> EdgePixels;
        EdgePixels edge_pixels;
        vector<EdgePixels> edge_pixels_vec;
        /** spatial coordinates of edge points (center of distribution) **/
        typedef vector<vector<double>> EdgePts;
        EdgePts edge_pts;
        vector<EdgePts> edge_pts_vec;
        /** mean position of the lidar pts in a specific pixel space **/
        typedef pcl::PointCloud<pcl::PointXYZI>::Ptr EdgeCloud; /** note: I is used to store the weight **/
        EdgeCloud edge_cloud;
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
                this -> OutputPath = ScenePath + "/outputs";
                this -> PcdsPath = ScenePath + "/pcds";
                this -> EdgeImgPath = ScenePath + "/edges/lidEdge.png";
                this -> ResultPath = ScenePath + "/results";
                this -> ProjPath = this -> OutputPath + "/byIntensity";
                this -> LidDensePcdPath = this -> OutputPath + "/lidDense" + to_string(numPcds) + ".pcd";
                this -> FlatImgPath = this -> ProjPath + "/flatLidarImage.bmp";
                this -> PolarPcdPath = this -> ProjPath + "/lidPolar.pcd";
                this -> CartPcdPath = this -> ProjPath + "/lidCartesian.pcd";
                this -> EdgeCheckImgPath = this -> ProjPath + "/edgeCheck.bmp";
                this -> TagsMapTxtPath = this -> ProjPath + "/tags_map.txt";
                this -> EdgeTxtPath = this -> OutputPath + "/lidEdgePix.txt";
                this -> EdgeOrgTxtPath = this -> OutputPath + +"/lid3dOut.txt";
                this -> EdgeTransTxtPath = this -> OutputPath + "/lidTrans.txt";
                this -> ParamsRecordPath = this -> OutputPath + "/ParamsRecord.txt";

                this -> LidPro2DPath = this -> OutputPath + "/lidPro2d.txt";
                this -> LidPro3DPath = this -> OutputPath + "/lidPro3d.txt";
            }
            string OutputPath;
            string PcdsPath;
            string EdgeImgPath;
            string ResultPath;
            string ProjPath;
            string LidDensePcdPath;
            string FlatImgPath;
            string PolarPcdPath;
            string CartPcdPath;
            string EdgeCheckImgPath;
            string TagsMapTxtPath;
            string EdgeTxtPath;
            string EdgeOrgTxtPath;
            string EdgeTransTxtPath;
            string ParamsRecordPath;
            string LidPro2DPath;
            string LidPro3DPath;
        };
        vector<struct SceneFilePath> scenesFilePath;

public:
        lidarProcess(string dataPath, bool byIntensity);
        /***** Point Cloud Generation *****/
        int ReadFileList(const std::string &folderPath, std::vector<std::string> &vFileList);
        void CreateDensePcd();
        void BagToPcd(string bagFile);


        /***** Edge Related *****/
        void EdgeToPixel();
        void ReadEdge();
        vector<vector<double>> EdgeCloudProjectToFisheye(vector<double> _p);
        void EdgePixCheck();
        vector<double> kdeFit(vector< vector <double> > edgePixels, int row_samples, int col_samples);


        /***** LiDAR Pre-Processing *****/
        std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> LidarToSphere();
        void SphereToPlaneRNN(pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolar, pcl::PointCloud<pcl::PointXYZI>::Ptr lidCartesian);
        void PixLookUp(pcl::PointCloud<pcl::PointXYZI>::Ptr lidCartesian);


        /***** Get and Set Methods *****/
        void setExtrinsic(vector<double> _p) {
            this->extrinsic.rx = _p[0];
            this->extrinsic.ry = _p[1];
            this->extrinsic.rz = _p[2];
            this->extrinsic.tx = _p[3];
            this->extrinsic.ty = _p[4];
            this->extrinsic.tz = _p[5];
        }

        void SetSceneIdx(int scene_idx) {
            this -> scene_idx = scene_idx;
        }
};
#endif