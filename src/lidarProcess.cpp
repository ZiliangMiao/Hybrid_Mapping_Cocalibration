// basic
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
// ros
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <std_msgs/Header.h>
// pcl
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <Eigen/Core>
// opencv
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
// include mlpack
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
// heading
#include "lidarProcess.h"

using namespace std;
using namespace cv;

// using namespace mlpack;
using namespace mlpack::kde;
using namespace mlpack::metric;
using namespace mlpack::tree;
using namespace mlpack::kernel;

using namespace arma;
using namespace Eigen;

lidarProcess::lidarProcess(string pkgPath, bool byIntensity)
{
    this -> byIntensity = byIntensity;
    struct ScenesPath scenesPath(pkgPath);
    struct SceneFilePath SC1(scenesPath.sc1);
    struct SceneFilePath SC2(scenesPath.sc2);
    struct SceneFilePath SC3(scenesPath.sc3);
    struct SceneFilePath SC4(scenesPath.sc4);
    struct SceneFilePath SC5(scenesPath.sc5);
    this -> scenesFilePath.push_back(SC1);
    this -> scenesFilePath.push_back(SC2);
    this -> scenesFilePath.push_back(SC3);
    this -> scenesFilePath.push_back(SC4);
    this -> scenesFilePath.push_back(SC5);
}

void lidarProcess::readEdge()
{
    string edgeOrgTxtPath = this -> scenesFilePath[this -> scIdx].EdgeOrgTxtPath;

    ifstream infile(edgeOrgTxtPath);
    string line;
    vector<vector<double>> lidPts;
    while (getline(infile, line))
    {
        stringstream ss(line);
        string tmp;
        vector<double> v;

        while (getline(ss, tmp, '\t')) // split string with "\t"
        {                           
            v.push_back(stod(tmp)); // string->double
        }
        if (v.size() == 3)
        {
            lidPts.push_back(v);
        }
    }

    ROS_ASSERT_MSG(lidPts.size() != 0, "LiDAR Read Edge Incorrect! Scene Index: %d", this -> numScenes);

    // Remove dumplicated points
    cout << "Imported LiDAR points: " << lidPts.size() << endl;
    std::sort(lidPts.begin(), lidPts.end());
    lidPts.erase(unique(lidPts.begin(), lidPts.end()), lidPts.end());

    // Construct pcl pointcloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr EdgeOrgCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ point;
    for (size_t i = 0; i < lidPts.size(); i++)
    {
        point.x = lidPts[i][0];
        point.y = lidPts[i][1];
        point.z = lidPts[i][2];
        EdgeOrgCloud->points.push_back(point);
    }
    cout << "Filtered LiDAR points: " << EdgeOrgCloud->points.size() << endl;
    this -> EdgeOrgCloud = EdgeOrgCloud;
}

std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> lidarProcess::lidarToSphere()
{
    double X, Y, Z;
    double radius;
    double theta, phi;
    double projProp;
    double thetaMin = M_PI, thetaMax = -M_PI;

    bool byIntensity = this->byIntensity;
    string lidDensePcdPath = this -> scenesFilePath[this -> scIdx].LidDensePcdPath;
    // original cartesian point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr lid3dOrg(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(lidDensePcdPath, *lid3dOrg);

    // check the original point cloud size
    int cloudSizeOrg = lid3dOrg->points.size();
    cout << "size of original cloud:" << cloudSizeOrg << endl;

    /********* PCL Filter - Pass Through *********/
    pcl::PassThrough<pcl::PointXYZI> disFlt;
    disFlt.setFilterFieldName("x");
    disFlt.setFilterLimits(-1e-3, 1e-3);
    disFlt.setNegative(true);
    disFlt.setInputCloud(lid3dOrg);
    disFlt.filter(*lid3dOrg);
    pcl::PassThrough<pcl::PointXYZI> disFlt1;
    disFlt.setFilterFieldName("y");
    disFlt.setFilterLimits(-1e-3, 1e-3);
    disFlt.setNegative(true);
    disFlt.setInputCloud(lid3dOrg);
    disFlt.filter(*lid3dOrg);

    // check the pass through filtered point cloud size
    int cloudSizeIntFlt = lid3dOrg->points.size();
    cout << "size of cloud after a pass through filter:" << cloudSizeIntFlt << endl;

    // statistical filter: outlier removal
    // pcl::PointCloud<pcl::PointXYZI>::Ptr lidStaFlt(new pcl::PointCloud<pcl::PointXYZI>);
    // pcl::StatisticalOutlierRemoval<pcl::PointXYZI> outlierFlt;
    // outlierFlt.setInputCloud(lidIntFlt);
    // outlierFlt.setMeanK(20);
    // outlierFlt.setStddevMulThresh(0.8);
    // outlierFlt.setNegative(false);
    // outlierFlt.filter(*lidStaFlt);

    // filter: radius outlier removal
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidStaFlt(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> outlierFlt;
    outlierFlt.setInputCloud(lid3dOrg);
    outlierFlt.setRadiusSearch(0.1);
    outlierFlt.setMinNeighborsInRadius(5);
    outlierFlt.setNegative(false);
    outlierFlt.filter(*lidStaFlt);

    // new container
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolar(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointXYZI pt_polar;
    pcl::PointXYZI pt_cartesian;

    // cloud size check
    int cloudSizeStaFlt = lidStaFlt->points.size();
    cout << "statistically filtered cloud size:" << cloudSizeStaFlt << endl;

    for (int i = 0; i < cloudSizeStaFlt; i++)
    {
        // assign the cartesian coordinate to pcl point cloud
        X = lidStaFlt->points[i].x;
        Y = lidStaFlt->points[i].y;
        Z = lidStaFlt->points[i].z;
        projProp = lidStaFlt->points[i].intensity;
        if (!byIntensity)
        {
            radius = projProp;
        }
        else
        {
            radius = sqrt(pow(X, 2) + pow(Y, 2) + pow(Z, 2));
        }

        // assign the polar coordinate to pcl point cloud
        phi = M_PI - atan2(Y, X);
        theta = acos(Z / radius);
        pt_polar.x = theta;
        pt_polar.y = phi;
        pt_polar.z = 0;
        pt_polar.intensity = projProp;
        lidPolar->points.push_back(pt_polar);

        if (theta > thetaMax)
        {
            thetaMax = theta;
        }
        if (theta < thetaMin)
        {
            thetaMin = theta;
        }
    }

    // output the important features of the point cloud
    cout << "polar cloud size:" << lidPolar->points.size() << endl;
    cout << "Min theta of lidar: " << thetaMin << endl;
    cout << "Max theta of lidar: " << thetaMax << endl;

    // save to pcd files and create tuple return
    string polarPcdPath = this -> scenesFilePath[this -> scIdx].PolarPcdPath;
    string cartPcdPath = this -> scenesFilePath[this -> scIdx].CartPcdPath;
    pcl::io::savePCDFileBinary(cartPcdPath, *lidStaFlt);
    pcl::io::savePCDFileBinary(polarPcdPath, *lidPolar);
    std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, pcl::PointCloud<pcl::PointXYZI>::Ptr> result;
    result = std::make_tuple(lidPolar, lidStaFlt);

    return result;
}

vector<vector<vector<int>>> lidarProcess::sphereToPlaneRNN(pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolar)
{
    double flatRows = this->flatRows;
    double flatCols = this->flatCols;

    struct Tags
    {
        int Label; /** label = 0 -> empty pixel; label = 1 -> normal pixel **/
        vector<int> Idx;
        vector<double> Mean(2);
        vector<double> Cov(2);
    };

    // define the data container
    cv::Mat flatImage = cv::Mat::zeros(flatRows, flatCols, CV_32FC1); // define the flat image
    vector<Tags> tagsList; // define the tag list
    vector<vector<Tags>> tagsMap(flatRows, vector<Tags>(flatCols));

    // construct kdtrees and load the point clouds
    // caution: the point cloud need to be setted before the loop
    pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
    kdtree.setInputCloud(lidPolar);

    // define the invalid search count
    int invalidSearch = 0; // search invalid count
    int invalidIndex = 0; // index invalid count
    double radPerPix = this -> radPerPix;
    double scale = 2;
    double searchRadius = scale * (radPerPix / 2);
    for (int u = 0; u < flatRows; ++u)
    {
        float theta_lb = u * radPerPix;
        float theta_ub = (u + 1) * radPerPix;
        float theta_center = (theta_ub + theta_lb) / 2;

        for (int v = 0; v < flatCols; ++v)
        {
            float phi_lb = v * radPerPix;       // upper bound of the current phi unit
            float phi_ub = (v + 1) * radPerPix; // upper bound of the current phi unit
            float phi_center = (phi_ub + phi_lb) / 2;

            // assign the theta and phi center to the searchPoint
            pcl::PointXYZI searchPoint;
            searchPoint.x = theta_center;
            searchPoint.y = phi_center;
            searchPoint.z = 0;

            // define the vector container for storing the info of searched points
            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance; // type of distance vector has to be float
            // use kdtree to search (radius search) the spherical point cloud
            int numRNN = kdtree.radiusSearch(searchPoint, searchRadius, pointIdxRadiusSearch, pointRadiusSquaredDistance); // number of the radius nearest neighbors
            if (numRNN == 0)                                                                                               // no point found
            {
                flatImage.at<float>(u, v) = 160; // intensity
                invalidSearch = invalidSearch + 1;
                // add tags
                tagsMap[u][v].Label = 0;
                tagsMap[u][v].Idx.push_back(0);
                tagsMap[u][v].Mean = 0;
                tagsMap[u][v].Std = 0;
            }
            else  // corresponding points are found in the radius neighborhood
            {
                double intensity = 0; // intensity channel
                vector<double> Theta;
                vector<double> Phi;
                for (int i = 0; i < pointIdxRadiusSearch.size(); ++i)
                {
                    if (pointIdxRadiusSearch[i] > lidPolar->points.size() - 1)
                    {
                        // caution: a bug is hidden here, index of the searched point is bigger than size of the whole point cloud
                        flatImage.at<float>(u, v) = 160; // intensity
                        invalidIndex = invalidIndex + 1;
                        continue;
                    }
                    intensity = intensity + (*lidPolar)[pointIdxRadiusSearch[i]].intensity;
                    Theta.push_bask((*lidPolar)[pointIdxRadiusSearch[i]].x);
                    Phi.push_back((*lidPolar)[pointIdxRadiusSearch[i]].y);
                    // add tags
                    tagsMap[u][v].Idx.push_back(pointIdxRadiusSearch[i]);
                }
                tagsMap[u][v].Label = 1;


                tagsMap[u][v].Mean[0] = 0;
                tagsMap[u][v].Mean[1] = 0;
                tagsMap[u][v].Cov[0] = 0;
                tagsMap[u][v].Cov[1] = 0;
                flatImage.at<float>(u, v) = intensity / numRNN; // intensity
            }
        }
    }

    string tagsMapTxtPath = this -> scenesFilePath[this -> scIdx].TagsMapTxtPath;
    ofstream outfile;
    outfile.open(tagsMapTxtPath, ios::out);
    if (!outfile.is_open())
    {
        cout << "Open file failure" << endl;
    }
    for (int u = 0; u < flatRows; ++u)
    {
        for (int v = 0; v < flatCols; ++v)
        {
            cout << tagsMap[u][v].size() << endl;
            for (int k = 0; k < tagsMap[u][v].size(); ++k) {
                /** k is the number of lidar points that the [u][v] pixel contains **/
                if (k == tagsMap[u][v].size() - 1) {
                    cout << tagsMap[u][v][k] << endl;
                    outfile << tagsMap[u][v][k] << "\t" << "*****" << "\t" << tagsMap[u][v].size() << endl;
                }
                else {
                    cout << tagsMap[u][v][k] << endl;
                    outfile << tagsMap[u][v][k] << "\t";
                }
            }
        }
    }



    cout << "number of invalid searches:" << invalidSearch << endl;
    cout << "number of invalid indices:" << invalidIndex << endl;
    string flatImgPath = this -> scenesFilePath[this -> scIdx].FlatImgPath;
    cout << "LiDAR flat image path: " << flatImgPath << endl;
    cv::imwrite(flatImgPath, flatImage);
    cout << "LiDAR flat image generated successfully! Scene Index: " << this -> scIdx << endl;
    return tagsMap;
}

vector<vector<double>> lidarProcess::edgeTransform()
{
    const double pix2rad = 4000 / (M_PI * 2);
    // declaration of point clouds and output vector
    pcl::PointCloud<pcl::PointXYZ>::Ptr _p_l(new pcl::PointCloud<pcl::PointXYZ>());
    vector<vector<double>> lidEdgePolar(2, vector<double>(this -> EdgeOrgCloud -> points.size()));

    // initialize the transformation matrix
    Eigen::Affine3d transMat = Eigen::Affine3d::Identity();

    // initialize the euler angle and transform it into rotation matrix
    Eigen::Vector3d eulerAngle(this->extrinsic.rz, this->extrinsic.ry, this->extrinsic.rx);
    Eigen::AngleAxisd xAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitX()));
    Eigen::AngleAxisd yAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
    Eigen::AngleAxisd zAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitZ()));
    Eigen::Matrix3d rot;
    rot = zAngle * yAngle * xAngle;
    transMat.rotate(rot);

    // initialize the translation vector
    transMat.translation() << this->extrinsic.tx, this->extrinsic.ty, this->extrinsic.tz;

    // point cloud rigid transformation
    pcl::transformPointCloud(*this -> EdgeOrgCloud, *_p_l, transMat);

    for (int i = 0; i < this -> EdgeOrgCloud -> points.size(); i++)
    {
        // assign the polar coordinate (theta, phi) to pcl point cloud
        lidEdgePolar[0][i] = pix2rad * acos(_p_l->points[i].z / sqrt(pow(_p_l->points[i].x, 2) + pow(_p_l->points[i].y, 2) + pow(_p_l->points[i].z, 2)));
        lidEdgePolar[1][i] = pix2rad * (M_PI - atan2(_p_l->points[i].y, _p_l->points[i].x));
    }

    return lidEdgePolar;
}

vector<vector<int>> lidarProcess::edgeToPixel()
{
    string edgeImgPath = this -> scenesFilePath[this -> scIdx].EdgeImgPath;
    cv::Mat edgeImage = cv::imread(edgeImgPath, cv::IMREAD_UNCHANGED);

    ROS_ASSERT_MSG(((edgeImage.rows != 0 && edgeImage.cols != 0) || (edgeImage.rows < 16384 || edgeImage.cols < 16384)), "Size of original fisheye image is 0, check the path and filename! Scene Index: %d", this -> numScenes);
    ROS_ASSERT_MSG((edgeImage.rows == this->flatRows || edgeImage.cols == this->flatCols), "Size of original fisheye image is incorrect! Scene Index: %d", this -> numScenes);

    vector<vector<int>> edgePixels;
    for (int u = 0; u < edgeImage.rows; ++u)
    {
        for (int v = 0; v < edgeImage.cols; ++v)
        {
            if (edgeImage.at<uchar>(u, v) > 127)
            {
                vector<int> pixel{u, v};
                edgePixels.push_back(pixel);
            }
        }
    }

    /********* write the coordinates into txt file *********/
    string edgeTxtPath = this -> scenesFilePath[this -> scIdx].EdgeTxtPath;
    ofstream outfile;
    outfile.open(edgeTxtPath, ios::out);
    if (!outfile.is_open())
    {
        cout << "Open file failure" << endl;
    }
    for (int i = 0; i < edgePixels.size(); ++i)
    {
        outfile << edgePixels[i][0] << "\t" << edgePixels[i][1] << endl;
    }
    outfile.close();

    return edgePixels;
}

void lidarProcess::edgePixCheck(vector<vector<int>> edgePixels)
{
    int flatCols = this->flatCols;
    int flatRows = this->flatRows;
    cv::Mat edgeCheck = cv::Mat::zeros(flatRows, flatCols, CV_8UC1);
    for (int i = 0; i < edgePixels.size(); ++i)
    {
        int u = edgePixels[i][0];
        int v = edgePixels[i][1];
        edgeCheck.at<uchar>(u, v) = 255;
    }
    string edgeCheckImgPath = this -> scenesFilePath[this -> scIdx].EdgeCheckImgPath;
    cv::imwrite(edgeCheckImgPath, edgeCheck);
}

void lidarProcess::pixLookUp(vector<vector<int>> edgePixels, vector<vector<vector<int>>> tagsMap, pcl::PointCloud<pcl::PointXYZI>::Ptr lidCartesian)
{

    pcl::PointCloud<pcl::PointXYZ>::Ptr lidEdgeOrg(new pcl::PointCloud<pcl::PointXYZ>());
    int invalidLookUp = 0;

    for (int i = 0; i < edgePixels.size(); ++i)
    {
        int u = edgePixels[i][0];
        int v = edgePixels[i][1];
        float X;
        float Y;
        float Z;
        int size = tagsMap[u][v].size();

        if (size == 0)
        {
            invalidLookUp = invalidLookUp + 1;
            X = 0;
            Y = 0;
            Z = 0;
            pcl::PointXYZ lidPt{X, Y, Z};
            lidEdgeOrg->points.push_back(lidPt);
        }
        else
        {
            X = 0;
            Y = 0;
            Z = 0;
            for (int j = 0; j < size; ++j)
            {
                int idx = tagsMap[u][v][j];
                pcl::PointXYZI pt = (*lidCartesian)[idx];
                X = X + pt.x;
                Y = Y + pt.y;
                Z = Z + pt.z;
            }
            // assign mean values to the output
            X = X / size;
            Y = Y / size;
            Z = Z / size;
            // vector<double> coordinates {X, Y, Z};
            // lidPixOrg.push_back(coordinates);
            pcl::PointXYZ lidPt{X, Y, Z};
            lidEdgeOrg->points.push_back(lidPt);
        }
    }

    cout << "number of invalid lookups(lidar): " << invalidLookUp << endl;
    this -> EdgeOrgCloud = lidEdgeOrg;

    // write the coordinates into txt file
    string edgeOrgTxtPath = this -> scenesFilePath[this -> scIdx].EdgeOrgTxtPath;
    ofstream outfile;
    outfile.open(edgeOrgTxtPath, ios::out);
    if (!outfile.is_open())
    {
        cout << "Open file failure" << endl;
    }
    // for(int i = 0; i < lidPixOrg.size(); ++i) {
    //     outfile << lidPixOrg[i][0] << "\t" << lidPixOrg[i][1] << "\t"  << lidPixOrg[i][2] << "\n" << endl;
    // }
    for (int i = 0; i < lidEdgeOrg->points.size(); ++i)
    {
        outfile << lidEdgeOrg->points[i].x
                << "\t" << lidEdgeOrg->points[i].y
                << "\t" << lidEdgeOrg->points[i].z << endl;
    }
    outfile.close();
}

// extrinsic and inverse intrinsic transform for visualization of lidar points in flat image
vector<vector<double>> lidarProcess::edgeVizTransform(vector<double> _p)
{
    Eigen::Matrix<double, 3, 1> eulerAngle(_p[0], _p[1], _p[2]);
    Eigen::Matrix<double, 3, 1> t{_p[3], _p[4], _p[5]};
    Eigen::Matrix<double, 2, 1> uv_0{_p[6], _p[7]};
    Eigen::Matrix<double, 6, 1> a_;
    switch (_p.size() - 3)
    {
        case 10:
            a_ << _p[8], _p[9], _p[10], _p[11], _p[12], double(0);
            break;
        case 9:
            a_ << _p[8], _p[9], double(0), _p[10], double(0), _p[11];
            break;
        default:
            a_ << double(0), _p[8], double(0), _p[9], double(0), _p[10];
            break;
    }

    double phi, theta;
    double inv_r, r;
    double res, val;

    // extrinsic transform
    Eigen::Matrix<double, 3, 3> R;
    Eigen::AngleAxisd xAngle(AngleAxisd(eulerAngle(0), Vector3d::UnitX()));
    Eigen::AngleAxisd yAngle(AngleAxisd(eulerAngle(1), Vector3d::UnitY()));
    Eigen::AngleAxisd zAngle(AngleAxisd(eulerAngle(2), Vector3d::UnitZ()));
    R = zAngle * yAngle * xAngle;

    Eigen::Matrix<double, 3, 1> p_;
    Eigen::Matrix<double, 3, 1> p_trans;
    Eigen::Matrix<double, 2, 1> S;
    Eigen::Matrix<double, 2, 1> p_uv;

    vector<vector<double>> lidProjection(2, vector<double>(this -> EdgeOrgCloud -> points.size()));

    for (int i = 0; i < this -> EdgeOrgCloud -> points.size(); i++)
    {
        p_ << this -> EdgeOrgCloud -> points[i].x, this -> EdgeOrgCloud -> points[i].y, this -> EdgeOrgCloud -> points[i].z;
        p_trans = R * p_ + t;
        // phi = atan2(p_trans(1), p_trans(0)) + M_PI;
        theta = acos(p_trans(2) / sqrt(pow(p_trans(0), 2) + pow(p_trans(1), 2) + pow(p_trans(2), 2)));
        inv_r = a_(0) + a_(1) * theta + a_(2) * pow(theta, 2) + a_(3) * pow(theta, 3) + a_(4) * pow(theta, 4) + a_(5) * pow(theta, 5);
        r = sqrt(p_trans(1) * p_trans(1) + p_trans(0) * p_trans(0));
        S = {-inv_r * p_trans(0) / r, -inv_r * p_trans(1) / r};
        p_uv = S + uv_0;
        lidProjection[0][i] = p_uv(0);
        lidProjection[1][i] = p_uv(1);
    }

    return lidProjection;
}

int lidarProcess::readFileList(const std::string &folderPath, std::vector<std::string> &vFileList)
{
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(folderPath.c_str())) == NULL)
    {
        return 0;
    }

    int num = 0;
    while ((dirp = readdir(dp)) != NULL)
    {
        std::string name = std::string(dirp->d_name);
        if (name != "." && name != "..")
        {
            vFileList.push_back(name);
            num++;
        }
    }
    closedir(dp);
    cout << "read file list success" << endl;

    return num;
}

void lidarProcess::bagToPcd(string bagFile)
{
    rosbag::Bag bag;
    bag.open(bagFile, rosbag::bagmode::Read);
    vector<string> topics;
    topics.push_back(string(this->topicName));
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    rosbag::View::iterator it = view.begin();
    pcl::PCLPointCloud2 pcl_pc2;
    pcl::PointCloud<pcl::PointXYZI>::Ptr intensityCloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; it != view.end(); it++, i++)
    {
        auto m = *it;
        sensor_msgs::PointCloud2::ConstPtr input = m.instantiate<sensor_msgs::PointCloud2>();
        pcl_conversions::toPCL(*input, pcl_pc2);
        pcl::fromPCLPointCloud2(pcl_pc2, *intensityCloud);
        string id_str = to_string(i);
        string pcdsPath = this -> scenesFilePath[this -> scIdx].PcdsPath;
        pcl::io::savePCDFileBinary(pcdsPath + "/" + id_str + ".pcd", *intensityCloud);
    }
}

bool lidarProcess::createDenseFile(){
    pcl::PCDReader reader; /** used for read PCD files **/
    vector <string> nameList;
    string pcdsPath = this -> scenesFilePath[this -> scIdx].PcdsPath;
    readFileList(pcdsPath, nameList);
    sort(nameList.begin(),nameList.end()); /** sort file names by order **/

    int groupSize = this -> numPcds; /** number of pcds to be merged **/
    int groupCount = nameList.size() / groupSize;

    // PCL PointCloud pointer. Remember that the pointer need to be given a new space
    pcl::PointCloud<pcl::PointXYZI>::Ptr input(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr output(new pcl::PointCloud<pcl::PointXYZI>);
    int outputId = 0;
    int nameLength = groupSize * groupCount;
    auto nameIter = nameList.begin();
    for(int i = 0; i < groupCount; i++){
        for(int j = 0; j < groupSize; j++){
            string fileName = pcdsPath + "/" + *nameIter;
            cout << fileName << endl;
            if(reader.read(fileName, *input) < 0){      // read PCD files, and save PointCloud in the pointer
                PCL_ERROR("File is not exist!");
                system("pause");
                return false;
            }
            int pointCount = input -> points.size();
            for(int k = 0; k < pointCount; k++){
                output -> points.push_back(input -> points[k]);
            }
            nameIter++;
        }
        string outputPath = this -> scenesFilePath[this -> scIdx].OutputPath;
        pcl::io::savePCDFileBinary(outputPath + "/lidDense" + to_string(groupSize) + ".pcd", *output);
        cout << "create dense file success" << endl;
    }
    return true;
}

void lidarProcess::calculateMaxIncidence()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr lidarDenseCloud(new pcl::PointCloud<pcl::PointXYZI>);
    string lidDensePcdPath = this -> scenesFilePath[this -> scIdx].LidDensePcdPath;
    pcl::io::loadPCDFile(lidDensePcdPath, *lidarDenseCloud);
    float theta;
    float radius;
    float x, y, z;
    int lidarCount = lidarDenseCloud->points.size();
    vector<float> Theta;
    for (int i = 0; i < lidarCount; i++)
    {
        x = lidarDenseCloud->points[i].x;
        y = lidarDenseCloud->points[i].y;
        z = lidarDenseCloud->points[i].z;
        radius = sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2));
        theta = asin(z / radius);
        Theta.push_back(theta);
    }
    sort(Theta.begin(), Theta.end());
    int j = 0;
    for (auto it = Theta.begin(); it != Theta.end(); it++)
    {
        if (*it > 0)
        {
            j++;
            cout << *it << endl;
        }
        if (j == 1000)
        {
            break;
        }
    }
}

// void lidarProcess::lidFlatImageShift(){
//     float pixPerDegree = this -> pixelPerDegree;
//     float camThetaMin = this -> camThetaMin;
//     float camThetaMax = this -> camThetaMax;
//     float lidThetaMin = this -> lidThetaMin;
//     float lidThetaMax = this -> lidThetaMax;
//     this -> imFlatRows = int((camThetaMax - camThetaMin) * pixPerDegree); // defined as the height(pixel) of the flat image // caution: the int division need a type coercion to float, otherwise the result would be zero
//     this -> imFlatCols = int(2 * M_PI * pixPerDegree); // defined as the width(pixel) of the flat image
//     int imFlatRows = this -> imFlatRows;
//     int imFlatCols = this -> imFlatCols;
//
//     cv::Mat lidImage = cv::imread(this -> flatLidarFile, cv::IMREAD_UNCHANGED);
//     try{
//         if (lidImage.rows != this->lidFlatRows || lidImage.cols != this->lidFlatCols){
//             throw "Error: size of flat image (LiDAR) is incorrect!";
//         }
//     }catch (const char* msg){
//         cerr << msg << endl;
//         abort();
//     }
//
//     cv::Mat lidImageShift = cv::Mat::zeros(imFlatRows, imFlatCols, CV_8UC1);
//     int lidMin = lidThetaMin * pixPerDegree;
//     for(int i = 0; i < imFlatRows; i++){
//         for(int j = 0; j < imFlatCols; j++){
//             if(i >= lidMin){
//                 lidImageShift.at<uchar>(i, j) = lidImage.at<uchar>(i - lidMin, j);
//             }
//             else {
//                 lidImageShift.at<uchar>(i, j) = 160;
//             }
//         }
//     }
//     cv::imwrite(this -> flatLidarShiftFile, lidImageShift);
// }

// void lidarProcess::sphereToPlaneKNN(std::tuple<pcl::PointCloud<pcl::PointXYZI>::Ptr, float, float> result){
//     //Input: spherical point cloud with RGB info (radius = 1073)
//     pcl::PointCloud<pcl::PointXYZI>::Ptr lidPolar;
//     float thetaMin;
//     float thetaMax;
//     std::tie(lidPolar, thetaMin, thetaMax) = result;

//     float pixPerDegree = this -> imageRadius / (M_PI / 2);
//     int flatHeight = int((thetaMax - thetaMin) * pixPerDegree); // defined as the height(pixel) of the flat image
//     // caution: the int division need a type coercion to float, otherwise the result would be zero
//     int flatWidth = int(2 * M_PI * pixPerDegree); // defined as the width(pixel) of the flat image
//     cv::Mat flatImage = cv::Mat::zeros(flatHeight, flatWidth, CV_32FC1); // define the flat image

//     // construct kdtrees and load the point clouds
//     pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
//     kdtree.setInputCloud(lidPolar);
//     pcl::KdTreeFLANN<pcl::PointXYZI> kdtree2;
//     kdtree2.setInputCloud(lidPolar);

//     // KNN search nnumber
//     int K = 1;

//     // X, Y are the pixel coordinates in the flat image
//     // use KDTree to search the spherical point cloud
//     for(int X = 0; X < flatWidth ; ++X){
//         float degreePerPix = (M_PI / 2) / this -> imageRadius;
//         // upper bound and lower bound of the current theta unit
//         float phi_lb = X * degreePerPix;
//         float phi_ub = (X+1) * degreePerPix;
//         float phi_center = (phi_ub + phi_lb) / 2;

//         for(int Y = 0; Y < flatHeight ; ++Y){
//             // upper bound and lower bound of the current phi unit
//             float theta_lb = thetaMin + Y * degreePerPix;
//             float theta_ub = thetaMin + (Y+1) * degreePerPix;
//             float theta_center = (theta_ub + theta_lb) / 2;

//             // assign the theta and phi center to the searchPoint
//             pcl::PointXYZI searchPoint;
//             searchPoint.x = phi_center;
//             searchPoint.y = theta_center;
//             searchPoint.z = 0;

//             // define the vector container for storing the info of searched points
//             std::vector<int> pointIdxSearch(K);
//             std::vector<float> pointSquaredDistance(K);

//             // k nearest neighbors search
//             kdtree.nearestKSearch(searchPoint, K, pointIdxSearch, pointSquaredDistance); // number of the radius nearest neighbors
//             flatImage.at<float>(Y, X) = (*lidPolar)[pointIdxSearch[0]].intensity; // intensity
//         }
//     }
//     cv::imwrite("flatLidarImageKNN.bmp", flatImage);
// }

// void lidarProcess::pp_callback(const pcl::visualization::PointPickingEvent& event, void *args)
// {
//     struct callback_args * data = (struct callback_args *)args;//点云数据 & 可视化窗口
//     if (event.getPointIndex() == -1)
//         return;
//     PointT current_point;
//     event.getPoint(current_point.x, current_point.y, current_point.z);
//     data->clicked_points_3d->points.push_back(current_point);
//     //Draw clicked points in red:
//     pcl::visualization::PointCloudColorHandlerCustom<PointT> red(data->clicked_points_3d, 255, 0, 0);
//     data->viewerPtr->removePointCloud("clicked_points");
//     data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
//     data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
//     std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;

// }

// bool lidarProcess::pointMark(){
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
//     pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("viewer"));

//     if (pcl::io::loadPCDFile(this -> lidarDenseFile, *cloud))
//     {
//         std::cerr << "ERROR: Cannot open file " << this -> lidarDenseFile << "! Aborting..." << std::endl;
//         return false;
//     }

//     std::cout << cloud->points.size() << std::endl;

//     cloud_mutex.lock();    // for not overwriting the point cloud

//     // Display pointcloud:
//     viewer->addPointCloud(cloud, "bunny_source");
//     viewer->setCameraPosition(0, 0, -2, 0, -1, 0, 0);

//     // Add point picking callback to viewer:
//     struct callback_args cb_args;
//     PointCloudT::Ptr clicked_points_3d(new PointCloudT);
//     cb_args.clicked_points_3d = clicked_points_3d;
//     cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(viewer);

//     viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);

//     std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;

//     // Spin until 'Q' is pressed:
//     viewer->spin();
//     std::cout << "done." << std::endl;

//     cloud_mutex.unlock();

//     while (!viewer->wasStopped())
//     {
//         viewer->spinOnce(100);
//         boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//     }

//     return true;
// }