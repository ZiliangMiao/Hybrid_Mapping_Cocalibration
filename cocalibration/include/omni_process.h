/** basic **/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <thread>
#include <time.h>
/** opencv **/
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
/** pcl **/
#include <pcl/common/common.h>
#include <Eigen/Core>
/** ros **/
#include <ros/ros.h>
#include <ros/package.h>
/** mlpack **/
#include <mlpack/core.hpp>
#include <mlpack/methods/kde/kde.hpp>
#include <mlpack/core/tree/octree.hpp>
#include <mlpack/core/tree/cover_tree.hpp>
/** headings **/
#include <define.h>
/** namespace **/
using namespace std;

class OmniProcess{
public:
    /** Essential Params **/
    cv::Mat cocalibImage;
    EdgeCloud::Ptr ocamEdgeCloud; // edge pixels
    Pair kImageSize = {2048, 2448};
    Pair kEffectiveRadius = {300, 1100};
    int kExcludeRadius = 200;
    /** File Directory Path **/
    int NUM_SPOT = 1;
    string DATASET_NAME;
    string PKG_PATH;
    string DATASET_PATH;
    string COCALIB_PATH;
    string EDGE_PATH;
    string RESULT_PATH;
    string PYSCRIPT_PATH; 

    string cocalibImagePath;
    string cocalibEdgeImagePath;
    string cocalibEdgeCloudPath;
    string cocalibKdePath;
    /***** Intrinsic Params *****/
    Int_D int_;

public:
    /** Funcs **/
    OmniProcess();
    void loadCocalibImage();
    void edgeExtraction();
    void generateEdgeCloud();
    std::vector<double> Kde(double bandwidth, double scale);
};
