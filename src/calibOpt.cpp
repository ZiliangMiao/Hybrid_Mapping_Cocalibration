//basic
#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
//ros
#include <visualization_msgs/MarkerArray.h>
//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//ceres
#include "ceres/ceres.h"
#include "glog/logging.h"
//eigen
#include <Eigen/Eigen>
//pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/vtk_lib_io.h> //loadPolygonFileOBJ
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>
//heading
#include "imageProcess.h"
#include "lidarProcess.h"

using namespace std;

typedef pcl::PointXYZINormal PointType;

const bool curvByInd = true
;
const bool curvBySea = !(curvByInd);

const bool curvatureViz = false;
const bool dbg_show_id = true;
ros::Publisher pub_curvature;

const bool featureViz = true;
const bool subRegionViz = false;
const bool calibViz = true;

const bool intensityWrite = true;
const bool curvatureWrite = true;

const string dataPath = "/home/godm/catkin_ws/src/Fisheye-LiDAR-Fusion/data_process/data/huiyuan2/";

/********* Define Containers *********/
int cloudSortInd[10000000];
int cloudLabel[10000000];
int cloudNeighborPicked[10000000];
float cloudCurvature[10000000];
float cloudIntensity[10000000];
pcl::PointCloud<PointType>::Ptr edgeFeaturePoints(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr edgeFeaturePointsAlter(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr flatFeaturePoints(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr flatFeaturePointsAlter(new pcl::PointCloud<PointType>);

bool compare(int i, int j) { return (cloudCurvature[i] <  cloudCurvature[j]); }

template <typename PointT>
void VisualizeCurvature(float *v_curv, int *v_label,
                        const pcl::PointCloud<PointT> &pcl_in) {
//    ROS_ASSERT(pcl_in.size() < 400000);

    /// Same marker attributes
    visualization_msgs::Marker txt_mk;
    txt_mk.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    txt_mk.ns = "default";
    txt_mk.id = 0;
    txt_mk.action = visualization_msgs::Marker::ADD;
    txt_mk.pose.orientation.x = 0;
    txt_mk.pose.orientation.y = 0;
    txt_mk.pose.orientation.z = 0;
    txt_mk.pose.orientation.w = 1;
    txt_mk.scale.z = 0.05;
    txt_mk.color.a = 1;
    txt_mk.color.r = 0;
    txt_mk.color.g = 1;
    txt_mk.color.b = 0;

    static visualization_msgs::MarkerArray curv_txt_msg;
    // for (size_t i = 0; i < curv_txt_msg.markers.size(); ++i) {
    //   auto &mk_i = curv_txt_msg.markers[i];
    //   mk_i = txt_mk;
    //   mk_i.header.stamp = txt_mk.header.stamp - ros::Duration(0.001);
    //   mk_i.action = visualization_msgs::Marker::DELETE;
    //   mk_i.text = "";
    //   mk_i.ns = "old";
    //   mk_i.color.a = 0;
    // }
    // pub_curvature.publish(curv_txt_msg);
    // ros::Rate r(200);
    // r.sleep();

    /// Marger array message
    static size_t pre_pt_num = 0;
    size_t pt_num = pcl_in.size();

    if (pre_pt_num == 0) {
        curv_txt_msg.markers.reserve(400000);
    }
    if (pre_pt_num > pt_num) {
        curv_txt_msg.markers.resize(pre_pt_num);
    } else {
        curv_txt_msg.markers.resize(pt_num);
    }

    int edge_num = 0, edgeless_num = 0, flat_num = 0, flatless_num = 0, nn = 0;

    /// Add marker and namespace
    for (size_t i = 0; i < pcl_in.size(); ++i) {
        auto curv = v_curv[i];
        auto label = v_label[i];  /// -1: flat, 0: less-flat, 1:less-edge, 2:edge
        const auto &pt = pcl_in[i];

        switch (label) {
            case 2: {
                /// edge
                auto &mk_i = curv_txt_msg.markers[i];
                mk_i = txt_mk;
                mk_i.ns = "edge";
                mk_i.id = i;
                mk_i.pose.position.x = pt.x;
                mk_i.pose.position.y = pt.y;
                mk_i.pose.position.z = pt.z;
                mk_i.color.a = 1;
                mk_i.color.r = 1;
                mk_i.color.g = 0;
                mk_i.color.b = 0;
                char cstr[10];
                snprintf(cstr, 9, "%.2f", curv);
                mk_i.text = std::string(cstr);
                /// debug
                if (dbg_show_id) {
                    mk_i.text = std::to_string(i);
                }

                edge_num++;
                break;
            }
            case 1: {
                /// less edge
                auto &mk_i = curv_txt_msg.markers[i];
                mk_i = txt_mk;
                mk_i.ns = "edgeless";
                mk_i.id = i;
                mk_i.pose.position.x = pt.x;
                mk_i.pose.position.y = pt.y;
                mk_i.pose.position.z = pt.z;
                mk_i.color.a = 0.5;
                mk_i.color.r = 0.5;
                mk_i.color.g = 0;
                mk_i.color.b = 0.8;
                char cstr[10];
                snprintf(cstr, 9, "%.2f", curv);
                mk_i.text = std::string(cstr);
                /// debug
                if (dbg_show_id) {
                    mk_i.text = std::to_string(i);
                }

                edgeless_num++;
                break;
            }
            case -1: {
                /// less flat
                auto &mk_i = curv_txt_msg.markers[i];
                mk_i = txt_mk;
                mk_i.ns = "flatless";
                mk_i.id = i;
                mk_i.pose.position.x = pt.x;
                mk_i.pose.position.y = pt.y;
                mk_i.pose.position.z = pt.z;
                mk_i.color.a = 0.5;
                mk_i.color.r = 0;
                mk_i.color.g = 0.5;
                mk_i.color.b = 0.8;
                char cstr[10];
                snprintf(cstr, 9, "%.2f", curv);
                mk_i.text = std::string(cstr);
                /// debug
                if (dbg_show_id) {
                    mk_i.text = std::to_string(i);
                }

                flatless_num++;
                break;
            }
            case -2: {
                /// flat
                auto &mk_i = curv_txt_msg.markers[i];
                mk_i = txt_mk;
                mk_i.ns = "flat";
                mk_i.id = i;
                mk_i.pose.position.x = pt.x;
                mk_i.pose.position.y = pt.y;
                mk_i.pose.position.z = pt.z;
                mk_i.color.a = 1;
                mk_i.color.r = 0;
                mk_i.color.g = 1;
                mk_i.color.b = 0;
                char cstr[10];
                snprintf(cstr, 9, "%.2f", curv);
                mk_i.text = std::string(cstr);
                /// debug
                if (dbg_show_id) {
                    mk_i.text = std::to_string(i);
                }

                flat_num++;
                break;
            }
            default: {
                /// Un-reliable
                /// Do nothing for label=99
                // ROS_ASSERT_MSG(false, "%d", label);
                auto &mk_i = curv_txt_msg.markers[i];
                mk_i = txt_mk;
                mk_i.ns = "unreliable";
                mk_i.id = i;
                mk_i.pose.position.x = pt.x;
                mk_i.pose.position.y = pt.y;
                mk_i.pose.position.z = pt.z;
                mk_i.color.a = 0;
                mk_i.color.r = 0;
                mk_i.color.g = 0;
                mk_i.color.b = 0;
                char cstr[10];
                snprintf(cstr, 9, "%.2f", curv);
                mk_i.text = std::string(cstr);

                nn++;
                break;
            }
        }
    }
    ROS_INFO("edge/edgeless/flatless/flat/nn num: [%d / %d / %d / %d / %d] - %lu",
             edge_num, edgeless_num, flatless_num, flat_num, nn, pt_num);

    /// Delete old points
    if (pre_pt_num > pt_num) {
        ROS_WARN("%lu > %lu", pre_pt_num, pt_num);
        // curv_txt_msg.markers.resize(pre_pt_num);
        for (size_t i = pt_num; i < pre_pt_num; ++i) {
            auto &mk_i = curv_txt_msg.markers[i];
            mk_i.action = visualization_msgs::Marker::DELETE;
            mk_i.color.a = 0;
            mk_i.color.r = 0;
            mk_i.color.g = 0;
            mk_i.color.b = 0;
            mk_i.ns = "old";
            mk_i.text = "";
        }
    }
    pre_pt_num = pt_num;

    pub_curvature.publish(curv_txt_msg);
}

void pclFilterTest() {
    pcl::PointCloud<PointType>::Ptr lidarCloudOrg(new pcl::PointCloud<PointType>);
    pcl::io::loadPCDFile(dataPath + "outputs/lidDense.pcd", *lidarCloudOrg);
    /********* PCL Filter - RNN *********/
    pcl::PointCloud<PointType>::Ptr lidStaFlt(new pcl::PointCloud<PointType>);
    pcl::RadiusOutlierRemoval<PointType> outlierFlt;
    outlierFlt.setInputCloud(lidarCloudOrg);
    cout << "cloud input is correct!" << endl;
    outlierFlt.setRadiusSearch(0.3);
    outlierFlt.setMinNeighborsInRadius(3);
    cout << "cloud set is correct!" << endl;
    outlierFlt.filter(*lidStaFlt);
    cout << "cloud filer is correct!" << endl;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer);
    pcl::visualization::PointCloudColorHandlerCustom<PointType> test_rgb(lidStaFlt, 0, 0, 255);
    viewer -> setBackgroundColor(0, 0, 0);
    viewer -> addPointCloud<PointType>(lidStaFlt, test_rgb, "edge feature cloud");
    viewer -> setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "edge feature cloud");
    while (!viewer -> wasStopped()) {
        viewer -> spin();
    }
}

pcl::PointCloud<PointType>::Ptr getModelCurvatures(pcl::PointCloud<PointType>::Ptr lidarCloudOrg)
{
    //计算法线--------------------------------------------------------------------------------------
    pcl::NormalEstimation<PointType, pcl::Normal> ne;
    ne.setInputCloud(lidarCloudOrg);
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    ne.setSearchMethod(tree); //设置搜索方法
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.05); //设置半径邻域搜索
//    ne.setKSearch(5);
    ne.compute(*cloud_normals); //计算法向量
    //计算曲率-------------------------------------------------------------------------------------
    pcl::PrincipalCurvaturesEstimation<PointType, pcl::Normal, pcl::PrincipalCurvatures>pc;
    pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr cloud_curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>);
    pc.setInputCloud(lidarCloudOrg);
    pc.setInputNormals(cloud_normals);
    pc.setSearchMethod(tree);
    pc.setRadiusSearch(0.05);
//    pc.setKSearch(5);
    pc.compute(*cloud_curvatures);

    float curvature = 0.0;
    for (int i = 0; i < cloud_curvatures -> size(); i++){
        //平均曲率
        curvature = ((*cloud_curvatures)[i].pc1 + (*cloud_curvatures)[i].pc2) / 2;
        //高斯曲率
//        curvature = (*cloud_curvatures)[i].pc1 * (*cloud_curvatures)[i].pc2;
        lidarCloudOrg -> points[i].curvature = curvature;
        cloudCurvature[i] = curvature;
    }
    return lidarCloudOrg;
}


/********* LIVOX_HORIZON_LOAM - LIVOX *********/
void lidarFeatureExtractor(lidarProcess lidarProcess) {
    pcl::PointCloud<PointType>::Ptr lidarCloudOrg(new pcl::PointCloud<PointType>);
    pcl::io::loadPCDFile(dataPath + "outputs/lidDense50.pcd", *lidarCloudOrg);

    /********* PCL Filter - RNN *********/
//    pcl::PointCloud<PointType>::Ptr lidStaFlt(new pcl::PointCloud<PointType>);
//    pcl::RadiusOutlierRemoval<PointType> outlierFlt;
//    outlierFlt.setInputCloud(lidarCloudOrg);
//    cout << "cloud input is correct!" << endl;
//    outlierFlt.setRadiusSearch(0.5);
//    outlierFlt.setMinNeighborsInRadius(3);
//    cout << "cloud set is correct!" << endl;
//    outlierFlt.filter(*lidStaFlt);
//    cout << "cloud filer is correct!" << endl;

    /********* PCL Filter - KNN *********/
//    pcl::PointCloud<PointType>::Ptr lidStaFlt(new pcl::PointCloud<PointType>);
//    pcl::StatisticalOutlierRemoval<PointType> outlierFlt;
//    outlierFlt.setInputCloud(lidarCloudOrg);
//    outlierFlt.setMeanK(20);
//    outlierFlt.setStddevMulThresh(0.8);
//    outlierFlt.filter(*lidStaFlt);

    /********* PCL Filter - Pass Through1 *********/
    pcl::PassThrough<PointType> disFlt;
    disFlt.setFilterFieldName("x");
    disFlt.setFilterLimits(-1e-3, 1e-3);
    disFlt.setNegative(true);
    disFlt.setInputCloud(lidarCloudOrg);
    disFlt.filter(*lidarCloudOrg);
    pcl::PassThrough<PointType> disFlt1;
    disFlt.setFilterFieldName("y");
    disFlt.setFilterLimits(-1e-3, 1e-3);
    disFlt.setNegative(true);
    disFlt.setInputCloud(lidarCloudOrg);
    disFlt.filter(*lidarCloudOrg);
//    pcl::PassThrough<PointType> disFlt2;
//    disFlt.setFilterFieldName("z");
//    disFlt.setFilterLimits(-55e-2, -15e-2);
//    disFlt.setNegative(true);
//    disFlt.setInputCloud(lidarCloudOrg);
//    disFlt.filter(*lidarCloudOrg);
//    pcl::PassThrough<PointType> disFlt3;
//    disFlt.setFilterFieldName("z");
//    disFlt.setFilterLimits(2.1, 5);
//    disFlt.setNegative(true);
//    disFlt.setInputCloud(lidarCloudOrg);
//    disFlt.filter(*lidarCloudOrg);
    /********* PCL Filter - Pass Through2 *********/
    pcl::PassThrough<PointType> disFlt4;
    disFlt.setFilterFieldName("x");
    disFlt.setFilterLimits(-10, 10);
    disFlt.setNegative(false);
    disFlt.setInputCloud(lidarCloudOrg);
    disFlt.filter(*lidarCloudOrg);
    pcl::PassThrough<PointType> disFlt5;
    disFlt.setFilterFieldName("y");
    disFlt.setFilterLimits(-10, 10);
    disFlt.setNegative(false);
    disFlt.setInputCloud(lidarCloudOrg);
    disFlt.filter(*lidarCloudOrg);

    /********* Define Parameters *********/
    int cloudSize = lidarCloudOrg -> points.size();
    cout << "efficient cloud size: " << cloudSize << endl;
    int numCurvSize = 5; /** number of neighbors to calculate the curvature **/
    int startInd = numCurvSize;
    int numRegion = 2000; /** number of sub regions **/
    int subRegionSize = int(cloudSize / numRegion) + 1;
    int numEdge = 2;
    int numFlat = 4;
    int numEdgeNeighbor = 5;
    int numFlatNeighbor = 5;
    int numAlterOptions = 20; /** alternative feature points **/
    float thresholdEdge = 0.05;
    float thresholdFlat = 0.02;
    float maxFeatureDis = 1e4;
    float minFeatureDis = 1e-4;
    float maxFeatureInt = 40; /** intensity is different from reflectivity **/
    float minFeatureInt = 7e-3; /** parameters are given by loam_livox **/

    int dbNum1 = 0;
    int dbNum2 = 0;
    int dbNum3 = 0;
    int dbNum4 = 0;
    int dbNum5 = 0; /** neighbor distance **/
    int invalidNum = 0;
    int invalidPts = 0;
    int numLabel1 = 0;
    int numLabel2 = 0;
    int numLabel3 = 0;
    int numLabel4 = 0;
    float maxDis = 0;

    /********* Compute Curvature By Searching *********/
    if (curvBySea) {
        lidarCloudOrg = getModelCurvatures(lidarCloudOrg);
    }

    for (int i = startInd; i < cloudSize - (startInd + 1); i++) {
        /********* Compute Distance By Original Index *********/
        float disFormer = sqrt(lidarCloudOrg -> points[i-1].x * lidarCloudOrg -> points[i-1].x +
                               lidarCloudOrg -> points[i-1].y * lidarCloudOrg -> points[i-1].y +
                               lidarCloudOrg -> points[i-1].z * lidarCloudOrg -> points[i-1].z);
        float dis = sqrt(lidarCloudOrg -> points[i].x * lidarCloudOrg -> points[i].x +
                         lidarCloudOrg -> points[i].y * lidarCloudOrg -> points[i].y +
                         lidarCloudOrg -> points[i].z * lidarCloudOrg -> points[i].z);
        float disLatter = sqrt(lidarCloudOrg -> points[i+1].x * lidarCloudOrg -> points[i+1].x +
                               lidarCloudOrg -> points[i+1].y * lidarCloudOrg -> points[i+1].y +
                               lidarCloudOrg -> points[i+1].z * lidarCloudOrg -> points[i+1].z);
        if (dis > maxDis) {
            maxDis = dis;
        }
        /********* Compute Intensity *********/
        float intensity = lidarCloudOrg -> points[i].intensity / (dis * dis);
        cloudIntensity[i] = intensity;
        /********* Compute Curvature By Original Index *********/
        if (curvByInd) {
            float diffX = 0, diffY = 0, diffZ = 0;
            for (int j = 1; j < numCurvSize; ++j) {
                diffX += lidarCloudOrg -> points[i - j].x + lidarCloudOrg -> points[i - j].x - 2 * lidarCloudOrg -> points[i].x;
                diffY += lidarCloudOrg -> points[i - j].y + lidarCloudOrg -> points[i - j].y - 2 * lidarCloudOrg -> points[i].y;
                diffZ += lidarCloudOrg -> points[i - j].z + lidarCloudOrg -> points[i - j].z - 2 * lidarCloudOrg -> points[i].z;
            }
            float curv = sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / ((2 * numCurvSize) * dis + 1e-3);
            lidarCloudOrg -> points[i].curvature = curv;
            cloudCurvature[i] = curv;
        }

        /********* Select Feature *********/
        cloudSortInd[i] = i; /** begin from ind = 5 **/
        cloudNeighborPicked[i] = 0; /** labeled 0 means neighbor points picked **/
        cloudLabel[i] = 0; /** labeled 0 means normal points **/
        /********* "Good Point" Filter *********/
        /********* 1. Distance *********/
        /********* livox_horizon_loam *********/
        if (fabs(dis) > maxFeatureDis || fabs(dis) < minFeatureDis || !std::isfinite(dis)) {
            cloudLabel[i] = 99; /** labeled 99 means the distance is invalid **/
            cloudNeighborPicked[i] = 1; /** labeled 1 means neighbor points are not picked **/
            dbNum1++;
            invalidNum++;
        }

        /********* 2.0. Neighbor Distance *********/
        /********* livox_horizon_loam *********/
        float neighborDiffXF = lidarCloudOrg -> points[i].x - lidarCloudOrg -> points[i - 1].x;
        float neighborDiffYF = lidarCloudOrg -> points[i].y - lidarCloudOrg -> points[i - 1].y;
        float neighborDiffZF = lidarCloudOrg -> points[i].z - lidarCloudOrg -> points[i - 1].z;
        float neighborDiffFormer = sqrt(neighborDiffXF * neighborDiffXF + neighborDiffYF * neighborDiffYF + neighborDiffZF * neighborDiffZF);
        float neighborDiffXL = lidarCloudOrg -> points[i + 1].x - lidarCloudOrg -> points[i].x;
        float neighborDiffYL = lidarCloudOrg -> points[i + 1].y - lidarCloudOrg -> points[i].y;
        float neighborDiffZL = lidarCloudOrg -> points[i + 1].z - lidarCloudOrg -> points[i].z;
        float neighborDiffLatter = sqrt(neighborDiffXL * neighborDiffXL + neighborDiffYL * neighborDiffYL + neighborDiffZL * neighborDiffZL);
        if (neighborDiffFormer > 0.05 * dis) {
            cloudNeighborPicked[i] = 1;
            cloudNeighborPicked[i - 1] = 1;
            dbNum5++;
        }
        else if (neighborDiffLatter > 0.05 * dis) {
            cloudNeighborPicked[i] = 1;
            cloudNeighborPicked[i + 1] = 1;
            dbNum5++;
        }

        /********* 2.1. Hidden Points *********/
        /********* loam_livox *********/
        neighborDiffFormer = sqrt(neighborDiffFormer);
        neighborDiffLatter = sqrt(neighborDiffLatter);
        if (disFormer > disLatter) {
            if (neighborDiffFormer > 0.1 * dis && dis > disFormer) {
                cloudNeighborPicked[i] = 1;
                dbNum2++;
                invalidNum++;
            }
        }
        else {
            if (neighborDiffLatter > 0.1 * dis && dis > disLatter) {
                cloudNeighborPicked[i] = 1;
                dbNum2++;
                invalidNum++;
            }
        }
        /********* 3. Intensity *********/
        /********* loam_livox *********/
        if (intensity > maxFeatureInt || intensity < minFeatureInt) {
            cloudLabel[i] = 88; /** labeled 88 means the intensity is invalid **/
            cloudNeighborPicked[i] = 1; /** labeled 1 means neighbor points are not picked **/
            dbNum3++;
            invalidNum++;
        }
    }

    /********* Write the Cloud Intensity to .txt File *********/
    ofstream outfile;
    if (intensityWrite) {
        outfile.open(dataPath + "outputs/intensity.txt", ios::out);
        if(!outfile.is_open ()){
            cout << "Open file failure" << endl;
        }
        for (int i = 0; i < cloudSize; i++) {
            outfile << cloudIntensity[i] << endl;
        }
        outfile.close();
    }
    /********* Write the Cloud Curvature to .txt File *********/
    if (curvatureWrite) {
        outfile.open(dataPath + "outputs/curvature.txt", ios::out);
        if(!outfile.is_open ()){
            cout << "Open file failure" << endl;
        }
        for (int i = 0; i < cloudSize; i++) {
            outfile << cloudCurvature[i] << endl;
        }
        outfile.close();
    }

    cout << "max distance of mid360: " << maxDis << endl;

    vector <pcl::PointCloud<PointType>::Ptr> subCloudVec;
    for (int i = 0; i < numRegion; i++) {
        int regionStartInd = startInd + subRegionSize * i;
        int regionEndInd = startInd + subRegionSize * (i + 1) - 1; /** not include the last point **/

        /********* Extract Points in SubRegions *********/
//        cout << "start index of the subregion: " << regionStartInd << " " << "end index of the subregion: " << regionEndInd << endl;
        pcl::PointCloud<PointType>::Ptr subCloud(new pcl::PointCloud<PointType>);

//        if (i == 0) {
//            for (int j = 0; j <= 10; j++) {
//                subCloud -> push_back(lidarCloudOrg -> points[j]);
//                subCloudVec.push_back(subCloud);
//            }
//        }
        for (int j = regionStartInd; j <= regionEndInd; j++) {
            subCloud -> push_back(lidarCloudOrg -> points[j]);
            subCloudVec.push_back(subCloud);
        }

        /********* Sort by Curvature (Small -> Large) *********/
        /** difference of two implementations need to be tested **/
        /** manual implementation - O(n^2) **/
//        for (int j = regionStartInd; j <= regionEndInd; j++) {
//            for (int k = j; k>= regionStartInd + 1; k--) {
//                if (cloudCurvature[cloudSortInd[k]] < cloudCurvature[cloudSortInd[k - 1]]) {
//                    int temp = cloudSortInd[k - 1];
//                    cloudSortInd[k - 1] = cloudSortInd[k];
//                    cloudSortInd[k] = temp;
//                }
//            }
//        }
        /** std sort implementation - O(n*logn) **/
        std::sort(cloudSortInd + regionStartInd, cloudSortInd + regionEndInd + 1, compare);

        /** determine whether the abnormal sort condition occurred **/
        for (int j = regionStartInd; j < regionEndInd - 1; ++j) {
            ROS_ASSERT_MSG(cloudCurvature[cloudSortInd[j]] <= cloudCurvature[cloudSortInd[j + 1]], "curvature sort failed!");
        }

        /** Max Curvature Filter **/
        float maxCurRegion = cloudCurvature[cloudSortInd[regionEndInd]];
        float sumCurRegion = 0.0; /** sum of curvatures except the max **/
        for (int j = regionEndInd - 1; j >= regionStartInd; j--) {
            sumCurRegion += cloudCurvature[cloudSortInd[j]];
        }
        if (maxCurRegion > sumCurRegion * 3) {
            cloudNeighborPicked[cloudSortInd[regionEndInd]] = 1;
            dbNum4++;
            invalidNum++;
        }

        /********* Select Feature - Edge *********/
        int numEdgeSelected = 0;
        for (int j = regionEndInd; j >= regionStartInd; j--) {
            int ind = cloudSortInd[j];
            if (cloudNeighborPicked[ind] != 0) continue;
            if (cloudCurvature[ind] > thresholdEdge) {
                numEdgeSelected ++;
                if (numEdgeSelected <= numEdge) {
                    cloudLabel[ind] = 2; /** labeled 2 means the point is an edge **/
                    edgeFeaturePoints -> push_back(lidarCloudOrg -> points[ind]);
                    edgeFeaturePointsAlter -> push_back(lidarCloudOrg -> points[ind]);
//                    cout << "index of region (edge feature): " << i << endl;
                    numLabel1++;
                }
                else if (numEdgeSelected <= numAlterOptions) {
                    cloudLabel[ind] = 1; /** labeled 1 means the point is an alternative edge **/
                    edgeFeaturePointsAlter -> push_back(lidarCloudOrg -> points[ind]);
                    numLabel2++;
                }
                else {
                    break;
                }
            }
        }
//        cout << "edge feature size: " << edgeFeaturePoints -> points.size() << endl;
//        cout << "alternative edge feature size: " << edgeFeaturePoints -> points.size() << endl;

        /********* Select Feature - Flat *********/
        int numFlatSelected = 0;
        for (int j = regionStartInd; j <= regionEndInd; j++) {
            int ind = cloudSortInd[j];
            if (cloudNeighborPicked[ind] != 0) continue;
            if (cloudCurvature[ind] < thresholdFlat) {
                numFlatSelected ++;
                if (numFlatSelected <= numEdge) {
                    cloudLabel[ind] = -2; /** labeled -2 means the point is an flat feature **/
                    flatFeaturePoints -> push_back(lidarCloudOrg -> points[ind]);
                    flatFeaturePointsAlter -> push_back(lidarCloudOrg -> points[ind]);
                    numLabel4++;
//                    cout << "index of region (flat feature): " << i << endl;
                }
                else if (numFlatSelected <= numAlterOptions) {
                    cloudLabel[ind] = -1; /** labeled -1 means the point is an alternative flat feature **/
                    flatFeaturePointsAlter -> push_back(lidarCloudOrg -> points[ind]);
                    numLabel3++;
                }
                else {
                    break;
                }
            }
        }
    }
    for (int i = 0; i < cloudSize; i++) {
        if (cloudNeighborPicked[i] != 0) {
            invalidPts++;
        }
    }

    cout << "**************************** Invalid Filter ****************************" << endl;
    cout << "number of invalid conditions (including 4 kind of points bellow): " << invalidNum << endl;
    cout << "number of invalid points: " << invalidPts << endl;
    cout << "number of max min distance: " << dbNum1 << endl;
    cout << "number of neighbor distance: " << dbNum5 << endl;
    cout << "number of hidden points: " << dbNum2 << endl;
    cout << "number of intensity: " << dbNum3 << endl;
    cout << "number of max curvature: " << dbNum4 << endl;
    cout << "**************************** Feature Numbers ****************************" << endl;
    cout << "number of edge features: " << numLabel1 << endl;
    cout << "number of alternative edge features: " << numLabel2 << endl;
    cout << "number of alternative flat features: " << numLabel3 << endl;
    cout << "number of flat features: " << numLabel4 << endl;

    /********* Write the Edge Feature to .txt File *********/
    outfile.open(dataPath + "outputs/lidEdgeFeature.txt", ios::out);
    if(!outfile.is_open ()){
        cout << "Open file failure" << endl;
    }
    for(int i = 0; i < edgeFeaturePoints -> points.size(); ++i) {
        outfile << edgeFeaturePoints -> points[i].x
                << "\t" << edgeFeaturePoints -> points[i].y
                << "\t" << edgeFeaturePoints -> points[i].z << endl;
    }
    outfile.close();

    /********* Visualization *********/
    if (curvatureViz) {
        VisualizeCurvature(cloudCurvature, cloudLabel, *lidarCloudOrg);
    }
    if (featureViz) {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer);
        pcl::visualization::PointCloudColorHandlerCustom<PointType> lid_rgb(lidarCloudOrg, 255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<PointType> feature_rgb(edgeFeaturePoints, 0, 0, 255);
        if (subRegionViz) {
            for (int i = 0; i < numRegion; i++) {
                pcl::visualization::PointCloudColorHandlerCustom<PointType> sub_rgb(subCloudVec[i], 0, 255, 0);
                viewer -> addPointCloud<PointType>(subCloudVec[i], sub_rgb, "sub cloud" + to_string(i));
                viewer -> setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sub cloud" + to_string(i));
            }
        }
        viewer -> setBackgroundColor(0, 0, 0);
        viewer -> addPointCloud<PointType>(lidarCloudOrg, lid_rgb, "origin lidar cloud");
        viewer -> addPointCloud<PointType>(edgeFeaturePoints, feature_rgb, "edge feature cloud");
        viewer -> setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "edge feature cloud");
        while (!viewer -> wasStopped()) {
            viewer -> spin();
        }
    }
}

void calibOpt(imageProcess imageProcess, lidarProcess lidarProcess) {
    imageProcess.readEdge();
    vector< vector <double> > camEdgePolar = imageProcess.edgeTransform();

    lidarProcess.extrinsic.rx = 0.001;
    lidarProcess.extrinsic.ry = 0.0197457;
    lidarProcess.extrinsic.rz = 0.13;
    lidarProcess.extrinsic.tx = 0.00891695;
    lidarProcess.extrinsic.ty = 0.00937508;
    lidarProcess.extrinsic.tz = 0.14;

    // 从这里开始新的起点~~最后造了孽~~~~~~~~
//    pcl::PointCloud<pcl::PointXYZ>::Ptr camEdgePts; 注意此处必须按照下方写法定义，否则会Assertion `px != 0' failed
    pcl::PointCloud<pcl::PointXYZ>::Ptr camEdgePts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr lidFeaturePts(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ pt_lid;
    for(int i = 0; i < edgeFeaturePoints -> points.size(); i++){
        double X = edgeFeaturePoints -> points[i].x;
        double Y = edgeFeaturePoints -> points[i].y;
        double Z = edgeFeaturePoints -> points[i].z;
        double radius = sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2));

        double phi = M_PI - atan2(Y, X);
        double theta = acos(Z / radius);

        pt_lid.x = phi;
        pt_lid.y = theta;
        pt_lid.z = 0;
        lidFeaturePts -> points.push_back(pt_lid);
    }

    pcl::PointXYZ pt_cam;
    for(int i = 0; i < camEdgePolar[0].size(); i++){
        double theta = camEdgePolar[0][i];
        double phi = camEdgePolar[1][i];

        pt_cam.x = phi;
        pt_cam.y = theta;
        pt_cam.z = 0;
        camEdgePts -> points.push_back(pt_cam);
    }

    cout << "number of camera edge points: " << camEdgePolar[0].size() << endl;

    if (calibViz) {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> lid_rgb(lidFeaturePts, 255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cam_rgb(camEdgePts, 0, 0, 255);
        viewer -> setBackgroundColor(0, 0, 0);
        viewer -> addPointCloud<pcl::PointXYZ>(lidFeaturePts, lid_rgb, "lidar feature cloud");
        viewer -> setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "lidar feature cloud");
        viewer -> addPointCloud<pcl::PointXYZ>(camEdgePts, cam_rgb, "camera edge cloud");

        while (!viewer -> wasStopped()) {
            viewer -> spin();
        }
    }
}

























/********* LOAM_LIVOX - HKU-MARS *********/
typedef pcl::PointXYZI PointType1;

enum E_point_type
{
    e_pt_normal = 0,                      // normal points
    e_pt_000 = 0x0001 << 0,               // points [0,0,0]
    e_pt_too_near = 0x0001 << 1,          // points in short range
    e_pt_reflectivity_low = 0x0001 << 2,  // low reflectivity
    e_pt_reflectivity_high = 0x0001 << 3, // high reflectivity
    e_pt_circle_edge = 0x0001 << 4,       // points near the edge of circle
    e_pt_nan = 0x0001 << 5,               // points with infinite value
    e_pt_small_view_angle = 0x0001 << 6,  // points with large viewed angle
};

enum E_feature_type // if and only if normal point can be labeled
{
    e_label_invalid = -1,
    e_label_unlabeled = 0,
    e_label_corner = 0x0001 << 0,
    e_label_surface = 0x0001 << 1,
    e_label_near_nan = 0x0001 << 2,
    e_label_near_zero = 0x0001 << 3,
    e_label_hight_intensity = 0x0001 << 4,
};

struct Pt_infos
{
    int   pt_type = e_pt_normal;
    int   pt_label = e_label_unlabeled;
    int   idx = 0.f;
    float raw_intensity = 0.f;
    float time_stamp = 0.0;
    float polar_angle = 0.f;
    int   polar_direction = 0;
    float polar_dis_sq2 = 0.f;
    float depth_sq2 = 0.f;
    float curvature = 0.0;
    float view_angle = 0.0;
    float sigma = 0.0;
    Eigen::Matrix< float, 2, 1 > pt_2d_img; // project to X==1 plane
};

struct Pt_compare
{
    //inline bool operator()( const pcl::PointXYZ& a,  const pcl::PointXYZ & b)
    template < typename _T >
    inline bool operator()( const _T &a, const _T &b )
    {
        return ( ( a.x < b.x ) || ( a.x == b.x && a.y < b.y ) || ( ( a.x == b.x ) && ( a.y == b.y ) && ( a.z < b.z ) ) );
    }

    template < typename _T >
    bool operator()( const _T &a, const _T &b ) const
    {
        return ( a.x == b.x ) && ( a.y == b.y ) && ( a.z == b.z );
    }
};

struct Pt_hasher
{
    template < typename _T >
    std::size_t operator()( const _T &k ) const
    {
        return ( ( std::hash< float >()( k.x ) ^ ( std::hash< float >()( k.y ) << 1 ) ) >> 1 ) ^ ( std::hash< float >()( k.z ) << 1 );
    }
};

/********* Define Container *********/
std::vector< Pt_infos >  m_pts_info_vec;
std::vector< PointType1 > m_raw_pts_vec;
std::unordered_map< PointType1, Pt_infos *, Pt_hasher, Pt_compare > m_map_pt_idx;
std::unordered_map< PointType1, Pt_infos *, Pt_hasher, Pt_compare >::iterator m_map_pt_idx_it;

/********* Define Parameters *********/
float m_input_points_size;
float m_time_internal_pts = 1.0e-5;
float m_max_edge_polar_pos = 0;
double m_current_time;
double m_last_maximum_time_stamp;
float thr_corner_curvature = 0.05;
float thr_surface_curvature = 0.01;
float minimum_view_angle = 10;

float m_livox_min_allow_dis = 1.0;
float m_livox_min_sigma = 7e-3;

void add_mask_of_point( Pt_infos *pt_infos, const E_point_type &pt_type, int neighbor_count = 0 )
{

    int idx = pt_infos->idx;
    pt_infos->pt_type |= pt_type;

    if ( neighbor_count > 0 )
    {
        for ( int i = -neighbor_count; i < neighbor_count; i++ )
        {
            idx = pt_infos->idx + i;

            if ( i != 0 && ( idx >= 0 ) && ( idx < ( int ) m_pts_info_vec.size() ) )
            {
                //screen_out << "Add mask, id  = " << idx << "  type = " << pt_type << endl;
                m_pts_info_vec[ idx ].pt_type |= pt_type;
            }
        }
    }
}

void eval_point( Pt_infos *pt_info )
{
    if ( pt_info->depth_sq2 < m_livox_min_allow_dis * m_livox_min_allow_dis ) // to close
    {
        //screen_out << "Add mask, id  = " << idx << "  type = e_too_near" << endl;
        add_mask_of_point( pt_info, e_pt_too_near );
    }

    pt_info->sigma = pt_info->raw_intensity / pt_info->polar_dis_sq2;

    if ( pt_info->sigma < m_livox_min_sigma )
    {
        //screen_out << "Add mask, id  = " << idx << "  type = e_reflectivity_low" << endl;
        add_mask_of_point( pt_info, e_pt_reflectivity_low );
    }
}

template < typename T >
T dis2_xy( T x, T y )
{
    return x * x + y * y;
}

template <typename T>
T vector_angle( const Eigen::Matrix<T, 3, 1> &vec_a, const Eigen::Matrix<T, 3, 1> &vec_b, int if_force_sharp_angle = 0 )
{
    T vec_a_norm = vec_a.norm();
    T vec_b_norm = vec_b.norm();
    if ( vec_a_norm == 0 || vec_b_norm == 0 ) // zero vector is pararrel to any vector.
    {
        return 0.0;
    }
    else
    {
        if ( if_force_sharp_angle )
        {
            // return acos( abs( vec_a.dot( vec_b ) )*0.9999 / ( vec_a_norm * vec_b_norm ) );
            return acos( abs( vec_a.dot( vec_b ) ) / ( vec_a_norm * vec_b_norm ) );
        }
        else
        {
            // return acos( (vec_a.dot(vec_b))*0.9999 / (vec_a_norm*vec_b_norm));
            return acos( ( vec_a.dot( vec_b ) ) / ( vec_a_norm * vec_b_norm ) );
        }
    }
}

template < typename T >
int projection_scan_3d_2d( pcl::PointCloud< T > &laserCloudIn, std::vector< float > &scan_id_index )
{

    unsigned int pts_size = laserCloudIn.size();
    m_pts_info_vec.clear();
    m_pts_info_vec.resize( pts_size );
    m_raw_pts_vec.resize( pts_size );
    std::vector< int > edge_idx;
    std::vector< int > split_idx;
    scan_id_index.resize( pts_size );
    m_map_pt_idx.clear();
    m_map_pt_idx.reserve( pts_size );
    std::vector< int > zero_idx;

    m_input_points_size = 0;

    for ( unsigned int idx = 0; idx < pts_size; idx++ )
    {
        m_raw_pts_vec[ idx ] = laserCloudIn.points[ idx ];
        Pt_infos *pt_info = &m_pts_info_vec[ idx ];
        m_map_pt_idx.insert( std::make_pair( laserCloudIn.points[ idx ], pt_info ) );
        pt_info->raw_intensity = laserCloudIn.points[ idx ].intensity;
        pt_info->idx = idx;
        pt_info->time_stamp = m_current_time + ( ( float ) idx ) * m_time_internal_pts;
        m_last_maximum_time_stamp = pt_info->time_stamp;
        m_input_points_size++;

        if ( !std::isfinite( laserCloudIn.points[ idx ].x ) ||
             !std::isfinite( laserCloudIn.points[ idx ].y ) ||
             !std::isfinite( laserCloudIn.points[ idx ].z ) )
        {
            add_mask_of_point( pt_info, e_pt_nan );
            continue;
        }

        if ( laserCloudIn.points[ idx ].x == 0 )
        {
            if ( idx == 0 )
            {
                // TODO: handle this case.
                cout << "First point should be normal!!!" << std::endl;

                pt_info->pt_2d_img << 0.01, 0.01;
                pt_info->polar_dis_sq2 = 0.0001;
                add_mask_of_point( pt_info, e_pt_000 );
                //return 0;
            }
            else
            {
                pt_info->pt_2d_img = m_pts_info_vec[ idx - 1 ].pt_2d_img;
                pt_info->polar_dis_sq2 = m_pts_info_vec[ idx - 1 ].polar_dis_sq2;
                add_mask_of_point( pt_info, e_pt_000 );
                continue;
            }
        }

        m_map_pt_idx.insert( std::make_pair( laserCloudIn.points[ idx ], pt_info ) );

        pt_info->depth_sq2 = depth2_xyz( laserCloudIn.points[ idx ].x, laserCloudIn.points[ idx ].y, laserCloudIn.points[ idx ].z );

        pt_info->pt_2d_img << laserCloudIn.points[ idx ].y / laserCloudIn.points[ idx ].x, laserCloudIn.points[ idx ].z / laserCloudIn.points[ idx ].x;
        pt_info->polar_dis_sq2 = dis2_xy( pt_info->pt_2d_img( 0 ), pt_info->pt_2d_img( 1 ) );

        eval_point( pt_info );

        if ( pt_info->polar_dis_sq2 > m_max_edge_polar_pos )
        {
            add_mask_of_point( pt_info, e_pt_circle_edge, 2 );
        }

        // Split scans
        if ( idx >= 1 )
        {
            float dis_incre = pt_info->polar_dis_sq2 - m_pts_info_vec[ idx - 1 ].polar_dis_sq2;

            if ( dis_incre > 0 ) // far away from zero
            {
                pt_info->polar_direction = 1;
            }

            if ( dis_incre < 0 ) // move toward zero
            {
                pt_info->polar_direction = -1;
            }

            if ( pt_info->polar_direction == -1 && m_pts_info_vec[ idx - 1 ].polar_direction == 1 )
            {
                if ( edge_idx.size() == 0 || ( idx - split_idx[ split_idx.size() - 1 ] ) > 50 )
                {
                    split_idx.push_back( idx );
                    edge_idx.push_back( idx );
                    continue;
                }
            }

            if ( pt_info->polar_direction == 1 && m_pts_info_vec[ idx - 1 ].polar_direction == -1 )
            {
                if ( zero_idx.size() == 0 || ( idx - split_idx[ split_idx.size() - 1 ] ) > 50 )
                {
                    split_idx.push_back( idx );

                    zero_idx.push_back( idx );
                    continue;
                }
            }
        }
    }
    split_idx.push_back( pts_size - 1 );

    int   val_index = 0;
    int   pt_angle_index = 0;
    float scan_angle = 0;
    int   internal_size = 0;

    if( split_idx.size() < 6) // minimum 3 petal of scan.
        return 0;

    for ( int idx = 0; idx < ( int ) pts_size; idx++ )
    {
        if ( val_index < split_idx.size() - 2 )
        {
            if ( idx == 0 || idx > split_idx[ val_index + 1 ] )
            {
                if ( idx > split_idx[ val_index + 1 ] )
                {
                    val_index++;
                }

                internal_size = split_idx[ val_index + 1 ] - split_idx[ val_index ];

                if ( m_pts_info_vec[ split_idx[ val_index + 1 ] ].polar_dis_sq2 > 10000 )
                {
                    pt_angle_index = split_idx[ val_index + 1 ] - ( int ) ( internal_size * 0.20 );
                    scan_angle = atan2( m_pts_info_vec[ pt_angle_index ].pt_2d_img( 1 ), m_pts_info_vec[ pt_angle_index ].pt_2d_img( 0 ) ) * 57.3;
                    scan_angle = scan_angle + 180.0;
                }
                else
                {
                    pt_angle_index = split_idx[ val_index + 1 ] - ( int ) ( internal_size * 0.80 );
                    scan_angle = atan2( m_pts_info_vec[ pt_angle_index ].pt_2d_img( 1 ), m_pts_info_vec[ pt_angle_index ].pt_2d_img( 0 ) ) * 57.3;
                    scan_angle = scan_angle + 180.0;
                }
            }
        }
        m_pts_info_vec[ idx ].polar_angle = scan_angle;
        scan_id_index[ idx ] = scan_angle;
    }

    return split_idx.size() - 1;
}

void compute_features()
{
    unsigned int pts_size = m_raw_pts_vec.size();
    size_t       curvature_ssd_size = 2;
    int          critical_rm_point = e_pt_000 | e_pt_nan;
    float        neighbor_accumulate_xyz[ 3 ] = { 0.0, 0.0, 0.0 };

    for ( size_t idx = curvature_ssd_size; idx < pts_size - curvature_ssd_size; idx++ )
    {
        if ( m_pts_info_vec[ idx ].pt_type & critical_rm_point )
        {
            continue;
        }

        /*********** Compute curvate ************/
        neighbor_accumulate_xyz[ 0 ] = 0.0;
        neighbor_accumulate_xyz[ 1 ] = 0.0;
        neighbor_accumulate_xyz[ 2 ] = 0.0;

        for ( size_t i = 1; i <= curvature_ssd_size; i++ )
        {
            if ( ( m_pts_info_vec[ idx + i ].pt_type & e_pt_000 ) || ( m_pts_info_vec[ idx - i ].pt_type & e_pt_000 ) )
            {
                if ( i == 1 )
                {
                    m_pts_info_vec[ idx ].pt_label |= e_label_near_zero;
                }
                else
                {
                    m_pts_info_vec[ idx ].pt_label = e_label_invalid;
                }
                break;
            }
            else if ( ( m_pts_info_vec[ idx + i ].pt_type & e_pt_nan ) || ( m_pts_info_vec[ idx - i ].pt_type & e_pt_nan ) )
            {
                if ( i == 1 )
                {
                    m_pts_info_vec[ idx ].pt_label |= e_label_near_nan;
                }
                else
                {
                    m_pts_info_vec[ idx ].pt_label = e_label_invalid;
                }
                break;
            }
            else
            {
                neighbor_accumulate_xyz[ 0 ] += m_raw_pts_vec[ idx + i ].x + m_raw_pts_vec[ idx - i ].x;
                neighbor_accumulate_xyz[ 1 ] += m_raw_pts_vec[ idx + i ].y + m_raw_pts_vec[ idx - i ].y;
                neighbor_accumulate_xyz[ 2 ] += m_raw_pts_vec[ idx + i ].z + m_raw_pts_vec[ idx - i ].z;
            }
        }

        if(m_pts_info_vec[ idx ].pt_label == e_label_invalid)
        {
            continue;
        }

        neighbor_accumulate_xyz[ 0 ] -= curvature_ssd_size * 2 * m_raw_pts_vec[ idx ].x;
        neighbor_accumulate_xyz[ 1 ] -= curvature_ssd_size * 2 * m_raw_pts_vec[ idx ].y;
        neighbor_accumulate_xyz[ 2 ] -= curvature_ssd_size * 2 * m_raw_pts_vec[ idx ].z;
        m_pts_info_vec[ idx ].curvature = neighbor_accumulate_xyz[ 0 ] * neighbor_accumulate_xyz[ 0 ] + neighbor_accumulate_xyz[ 1 ] * neighbor_accumulate_xyz[ 1 ] +
                                          neighbor_accumulate_xyz[ 2 ] * neighbor_accumulate_xyz[ 2 ];

        /*********** Compute plane angle ************/
        Eigen::Matrix< float, 3, 1 > vec_a( m_raw_pts_vec[ idx ].x, m_raw_pts_vec[ idx ].y, m_raw_pts_vec[ idx ].z );
        Eigen::Matrix< float, 3, 1 > vec_b( m_raw_pts_vec[ idx + curvature_ssd_size ].x - m_raw_pts_vec[ idx - curvature_ssd_size ].x,
                                            m_raw_pts_vec[ idx + curvature_ssd_size ].y - m_raw_pts_vec[ idx - curvature_ssd_size ].y,
                                            m_raw_pts_vec[ idx + curvature_ssd_size ].z - m_raw_pts_vec[ idx - curvature_ssd_size ].z );
        m_pts_info_vec[ idx ].view_angle = vector_angle( vec_a  , vec_b, 1 ) * 57.3;

        //printf( "Idx = %d, angle = %.2f\r\n", idx,  m_pts_info_vec[ idx ].view_angle );
        if ( m_pts_info_vec[ idx ].view_angle > minimum_view_angle )
        {

            if( m_pts_info_vec[ idx ].curvature < thr_surface_curvature )
            {
                m_pts_info_vec[ idx ].pt_label |= e_label_surface;
            }

            float sq2_diff = 0.1;

            if ( m_pts_info_vec[ idx ].curvature > thr_corner_curvature )
            {
                if ( m_pts_info_vec[ idx ].depth_sq2 <= m_pts_info_vec[ idx - curvature_ssd_size ].depth_sq2 &&
                     m_pts_info_vec[ idx ].depth_sq2 <= m_pts_info_vec[ idx + curvature_ssd_size ].depth_sq2 )
                {
                    if ( abs( m_pts_info_vec[ idx ].depth_sq2 - m_pts_info_vec[ idx - curvature_ssd_size ].depth_sq2 ) < sq2_diff * m_pts_info_vec[ idx ].depth_sq2 ||
                         abs( m_pts_info_vec[ idx ].depth_sq2 - m_pts_info_vec[ idx + curvature_ssd_size ].depth_sq2 ) < sq2_diff * m_pts_info_vec[ idx ].depth_sq2 )
                        m_pts_info_vec[ idx ].pt_label |= e_label_corner;
                }
            }
        }
    }
}