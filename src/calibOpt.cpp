//basic
#include <iostream>
#include <algorithm>
#include <vector>
#include <math.h>
//ros
#include <visualization_msgs/MarkerArray.h>
#include <ros/package.h>
//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
//ceres
#include "ceres/ceres.h"
#include "glog/logging.h"
//pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
//heading
#include "imageProcess.h"
#include "lidarProcess.h"

using namespace std;

typedef pcl::PointXYZINormal PointType;

const bool curvatureViz = false;
const bool dbg_show_id = true;
ros::Publisher pub_curvature;

const bool featureViz = true;
const bool calibViz = false;

string getDataPath(){
    std::string currPkgDir = ros::package::getPath("data_process");
    std::string dataPath = currPkgDir + "/data/runYangIn/";
    return dataPath;
}

const string dataPath = getDataPath();

/********* Define Containers *********/
int cloudSortInd[10000000];
int cloudLabel[10000000];
int cloudNeighborPicked[10000000];
float cloudCurvature[10000000];
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
    pcl::io::loadPCDFile(dataPath + "/outputs/lidDense.pcd", *lidarCloudOrg);
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


void lidarFeatureExtractor(lidarProcess lidarProcess) {
    pcl::PointCloud<PointType>::Ptr lidarCloudOrg(new pcl::PointCloud<PointType>);
    pcl::io::loadPCDFile( dataPath + "/outputs/lidDense150.pcd", *lidarCloudOrg);

    /********* PCL Filter - RNN *********/
    pcl::PointCloud<PointType>::Ptr lidStaFlt(new pcl::PointCloud<PointType>);
    pcl::RadiusOutlierRemoval<PointType> outlierFlt;
    outlierFlt.setInputCloud(lidarCloudOrg);
    cout << "cloud input is correct!" << endl;
    outlierFlt.setRadiusSearch(0.5);
    outlierFlt.setMinNeighborsInRadius(3);
    cout << "cloud set is correct!" << endl;
    outlierFlt.filter(*lidStaFlt);
    cout << "cloud filer is correct!" << endl;

    /********* PCL Filter - RNN *********/
//    pcl::PointCloud<PointType>::Ptr lidStaFlt(new pcl::PointCloud<PointType>);
//    pcl::StatisticalOutlierRemoval<PointType> outlierFlt;
//    outlierFlt.setInputCloud(lidarCloudOrg);
//    outlierFlt.setMeanK(20);
//    outlierFlt.setStddevMulThresh(0.8);
//    outlierFlt.filter(*lidStaFlt);


    /********* Define Parameters *********/
    int cloudSize = lidarCloudOrg -> points.size();
    cout << "original cloud size: " << cloudSize << endl;
    int numCurvSize = 5; /** number of neighbors to calculate the curvature **/
    int startInd = numCurvSize;
    int numRegion = 1000; /** number of sub regions **/
    int subRegionSize = int(cloudSize / numRegion) + 1;
    int numEdge = 2;
    int numFlat = 4;
    int numEdgeNeighbor = 5;
    int numFlatNeighbor = 5;
    int numAlterOptions = 20; /** alternative feature points **/
    float thresholdEdge = 0.01;
    float thresholdFlat = 0.01;
    float maxFeatureDis = 1e4;
    float minFeatureDis = 1e-4;
    float maxFeatureInt = 1e-1; /** intensity is different from reflectivity **/
    float minFeatureInt = 7e-3; /** parameters are given by loam_livox **/

    for (int i = startInd; i < cloudSize - (startInd + 1); i++) {
        /********* Compute Distance *********/
        float disFormer = sqrt(lidarCloudOrg -> points[i-1].x * lidarCloudOrg -> points[i-1].x +
                               lidarCloudOrg -> points[i-1].y * lidarCloudOrg -> points[i-1].y +
                               lidarCloudOrg -> points[i-1].z * lidarCloudOrg -> points[i-1].z);
        float dis = sqrt(lidarCloudOrg -> points[i].x * lidarCloudOrg -> points[i].x +
                         lidarCloudOrg -> points[i].y * lidarCloudOrg -> points[i].y +
                         lidarCloudOrg -> points[i].z * lidarCloudOrg -> points[i].z);
        float disLatter = sqrt(lidarCloudOrg -> points[i+1].x * lidarCloudOrg -> points[i+1].x +
                               lidarCloudOrg -> points[i+1].y * lidarCloudOrg -> points[i+1].y +
                               lidarCloudOrg -> points[i+1].z * lidarCloudOrg -> points[i+1].z);
//        cout << dis << endl;
        /********* Compute Intensity *********/
        float intensity = lidarCloudOrg -> points[i].intensity / (dis * dis);
        /********* Compute Curvature *********/
        float diffX = 0, diffY = 0, diffZ = 0;
        for (int j = 1; j < numCurvSize; ++j) {
            diffX += lidarCloudOrg -> points[i - j].x + lidarCloudOrg -> points[i - j].x - 2 * lidarCloudOrg -> points[i].x;
            diffY += lidarCloudOrg -> points[i - j].y + lidarCloudOrg -> points[i - j].y - 2 * lidarCloudOrg -> points[i].y;
            diffZ += lidarCloudOrg -> points[i - j].z + lidarCloudOrg -> points[i - j].z - 2 * lidarCloudOrg -> points[i].z;
        }
        float curv = sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / ((2 * numCurvSize) * dis + 1e-3);
//        cout << curv << endl;
        lidarCloudOrg -> points[i].curvature = curv;
        cloudCurvature[i] = curv;

        /********* Select Feature *********/
        cloudSortInd[i] = i; /** begin from ind = 5 **/
        cloudNeighborPicked[i] = 0; /** labeled 0 means neighbor points picked **/
        cloudLabel[i] = 0; /** labeled 0 means normal points **/
        /********* "Good Point" Filter *********/
        /********* 1. Distance *********/
        /********* livox_horizon_loam *********/
        if (fabs(dis) > maxFeatureDis || fabs(dis) < minFeatureDis || !std::isfinite(dis)) {
//            cout << (fabs(dis) > maxFeatureDis) << " " << (fabs(dis) < minFeatureDis) << " " << !std::isfinite(dis) << endl;
            cloudLabel[i] = 99; /** labeled 99 means the distance is invalid **/
            cloudNeighborPicked[i] = 1; /** labeled 1 means neighbor points are not picked **/
        };

        /********* 2.0. Neighbor Distance *********/
        /********* livox_horizon_loam *********/
        float neighborDiffXF = lidarCloudOrg -> points[i].x - lidarCloudOrg -> points[i - 1].x;
        float neighborDiffYF = lidarCloudOrg -> points[i].y - lidarCloudOrg -> points[i - 1].y;
        float neighborDiffZF = lidarCloudOrg -> points[i].z - lidarCloudOrg -> points[i - 1].z;
        float neighborDiffFormer = neighborDiffXF * neighborDiffXF + neighborDiffYF * neighborDiffYF + neighborDiffZF * neighborDiffZF;
        float neighborDiffXL = lidarCloudOrg -> points[i + 1].x - lidarCloudOrg -> points[i].x;
        float neighborDiffYL = lidarCloudOrg -> points[i + 1].y - lidarCloudOrg -> points[i].y;
        float neighborDiffZL = lidarCloudOrg -> points[i + 1].z - lidarCloudOrg -> points[i].z;
        float neighborDiffLatter = neighborDiffXL * neighborDiffXL + neighborDiffYL * neighborDiffYL + neighborDiffZL * neighborDiffZL;
//
//        cout << neighborDiffFormer << " " << neighborDiffLatter << " " << 0.00015 * dis * dis << endl;
//
//        if (neighborDiffFormer > 0.00015 * dis * dis && neighborDiffLatter > 0.00015 * dis * dis) {
//            cloudNeighborPicked[i] = 1;
//        }
        /********* 2.1. Neighbor Distance *********/
        /********* loam_livox *********/
        neighborDiffFormer = sqrt(neighborDiffFormer);
        neighborDiffLatter = sqrt(neighborDiffLatter);
        if (disFormer > disLatter) {
            if (neighborDiffFormer > 0.1 * dis && dis > disFormer) {
                cloudNeighborPicked[i] = 1;
            }
        }
        else {
            if (neighborDiffLatter > 0.1 * dis && dis > disLatter) {
                cloudNeighborPicked[i] = 1;
            }
        }
        /********* 3. Intensity *********/
        /********* loam_livox *********/
        if (intensity > maxFeatureInt || intensity < minFeatureInt) {
            cloudLabel[i] = 88; /** labeled 88 means the intensity is invalid **/
            cloudNeighborPicked[i] = 1; /** labeled 1 means neighbor points are not picked **/
        }
    }

    for (int i = 0; i < numRegion; i++) {
        int regionStartInd = startInd + subRegionSize * i;
        int regionEndInd = startInd + subRegionSize * (i + 1) - 1; /** not include the last point **/

        /********* Sort by Curvature (Small -> Large) *********/
        /** difference of two implementations need to be tested **/
        /** manual implementation - O(n^2) **/
        for (int j = regionStartInd; j <= regionEndInd; j++) {
            for (int k = j; k>= regionStartInd + 1; k--) {
                if (cloudCurvature[cloudSortInd[k]] < cloudCurvature[cloudSortInd[k - 1]]) {
                    int temp = cloudSortInd[k - 1];
                    cloudSortInd[k - 1] = cloudSortInd[k];
                    cloudSortInd[k] = temp;
                }
            }
        }
        /** std sort implementation - O(n*logn) **/
//        std::sort(cloudSortInd + regionStartInd, cloudSortInd + regionEndInd + 1, compare);

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
                    cloudNeighborPicked[ind] = 1;
                }
                else if (numEdgeSelected <= numAlterOptions) {
                    cloudLabel[ind] = 1; /** labeled 1 means the point is an alternative edge **/
                    edgeFeaturePointsAlter -> push_back(lidarCloudOrg -> points[ind]);
                    cloudNeighborPicked[ind] = 1;
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
                    cloudNeighborPicked[ind] = 1;
                }
                else if (numFlatSelected <= numAlterOptions) {
                    cloudLabel[ind] = -1; /** labeled -1 means the point is an alternative flat feature **/
                    flatFeaturePointsAlter -> push_back(lidarCloudOrg -> points[ind]);
                    cloudNeighborPicked[ind] = 1;
                }
                else {
                    break;
                }
            }
        }
    }

//    cout << "flat feature size: " << flatFeaturePoints -> points.size() << endl;
//    cout << "alternative flat feature size: " << flatFeaturePoints -> points.size() << endl;

    int numValid = 0;
    int numLabel1 = 0;
    int numLabel2 = 0;
    int numLabel3 = 0;
    int numLabel4 = 0;
    for (int i = startInd; i < cloudSize - (startInd + 1); i++) {
        if (cloudNeighborPicked[i] == 1) {
            numValid++;
        }
        if (cloudLabel[i] == -2) {
            numLabel1++;
        }
        if (cloudLabel[i] == -1) {
            numLabel2++;
        }
        if (cloudLabel[i] == 1) {
            numLabel3++;
        }
        if (cloudLabel[i] == 2) {
            numLabel4++;
        }
    }
    cout << "number of valid points: " << numValid << endl;
    cout << "number of flat features: " << numLabel1 << endl;
    cout << "number of alternative flat features: " << numLabel2 << endl;
    cout << "number of alternative edge features: " << numLabel3 << endl;
    cout << "number of edge features: " << numLabel4 << endl;


    if (curvatureViz) {
        VisualizeCurvature(cloudCurvature, cloudLabel, *lidarCloudOrg);
    }

    if (featureViz) {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer);
        pcl::visualization::PointCloudColorHandlerCustom<PointType> lid_rgb(lidarCloudOrg, 255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<PointType> feature_rgb(edgeFeaturePoints, 0, 0, 255);
        viewer -> setBackgroundColor(0, 0, 0);
        viewer -> addPointCloud<PointType>(lidarCloudOrg, lid_rgb, "origin lidar cloud");
        viewer -> addPointCloud<PointType>(edgeFeaturePoints, feature_rgb, "edge feature cloud");
        viewer -> setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "edge feature cloud");
        while (!viewer -> wasStopped()) {
            viewer -> spin();
        }
    }
}


void calibOpt(imageProcess imageProcess, lidarProcess lidarProcess) {
    imageProcess.readEdge();
    vector< vector <double> > camEdgeOrg = imageProcess.camEdgeOrg;
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

//    pcl::PointXYZ pt_lid;
//    for(int i = 0; i < lidFeaturePolar[0].size(); i++){
//        if (i % 30 == 0){
//            double theta = lidFeaturePolar[0][i];
//            double phi = lidFeaturePolar[1][i];
//
//            pt_lid.x = phi;
//            pt_lid.y = theta;
//            pt_lid.z = 0;
//            lidFeaturePts -> points.push_back(pt_lid);
//        }
//    }

//    cout << "number of lidar feature points: " << lidFeaturePts -> points.size() << endl;

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

    if (calibViz == true) {
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