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
        lidarProcess(string dataPath, bool byIntensity);
        
        void readEdge();
        void bagToPcd(string bagFile);
        bool createDenseFile();
        void calculateMaxIncidence();

        int readFileList(const std::string &folderPath, std::vector<std::string> &vFileList);

        std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> lidarToSphere();
        vector< vector< vector<int> > > sphereToPlaneRNN(pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolar);
        vector< vector<double> > edgeTransform();
        vector<vector<double>> edgeVizTransform(vector<double> _p);
        vector< vector <int> > edgeToPixel();
        void pixLookUp(vector< vector <int> > edgePixels, vector< vector< vector<int> > > tagsMap, pcl::PointCloud<pcl::PointXYZI>::Ptr lidCartesian);
        void edgePixCheck(vector< vector<int> > edgePixels);
        vector<double> kdeFit(vector< vector <double> > edgePixels, int row_samples, int col_samples);
        void setImageAngle(float camThetaMin, float camThetaMax);
        // void lidFlatImageShift();

    public:
        bool byIntensity = true;
        static const int numPcds = 500;

        int flatRows = int((double)110/90 * 1000) + 1;
        int flatCols = 4000;
        double radPerPix = (M_PI/2) / 1000;
        string topicName = "/livox/lidar";
        pcl::PointCloud<pcl::PointXYZ>::Ptr EdgeOrgCloud;

        struct Extrinsic {
            double rx = 0;
            double ry = 0;
            double rz = 0;
            double tx = 0;
            double ty = 0;
            double tz = 0;
        } extrinsic;

        void setExtrinsic(vector<double> _p) {
            this->extrinsic.rx = _p[0];
            this->extrinsic.ry = _p[1];
            this->extrinsic.rz = _p[2];
            this->extrinsic.tx = _p[3];
            this->extrinsic.ty = _p[4];
            this->extrinsic.tz = _p[5];
        }

        /********* 所有Path定义中的/留前不留后 *********/
        /********* Data Path of Multi-Scenes *********/
        int scIdx = 0;
        void setSceneIdx(int scIdx) {
            this -> scIdx = scIdx;
        }

        static const int numScenes = 5;
        struct ScenesPath
        {
            ScenesPath(string pkgPath) {
                this -> sc1 = pkgPath + "/data/runYangIn";
                this -> sc2 = pkgPath + "/data/huiyuan2";
                this -> sc3 = pkgPath + "/data/12";
                this -> sc4 = pkgPath + "/data/conferenceF2-P1";
                this -> sc5 = pkgPath + "/data/conferenceF2-P2";
            }
            string sc1;
            string sc2;
            string sc3;
            string sc4;
            string sc5;
        };

        /********* File Path of the Specific Scene *********/
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
                this -> TagsMapTxtPath = this -> ProjPath + "/tagsMap.txt";
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
};
#endif