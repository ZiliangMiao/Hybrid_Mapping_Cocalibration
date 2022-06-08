#ifndef _FEATUREE_H
#define _FEATUREE_H

class feature{
    public:
        feature(string dataPath, bool byIntensity);
        void showCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2);
        void rotate(int n);
        void visualRotate();
        float calculateDist(pcl::PointCloud<pcl::PointXYZ>::Ptr imageCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr lidarCloud);
        // void poseCalculation(vector<Point2f> points1, vector<Point2f> points2);    
        // void txtToClousd();
        void roughMatch(int degree, int x_offset, int y_offset);
        void imageLongitudeAndLatitude();
        void lidarLongitudeAndLatitude();
        void visualComparism();
    public:
        bool byIntensity = true;
        pcl::PointCloud<pcl::PointXYZ>::Ptr imageCloud;
        pcl::PointCloud<pcl::PointXYZ>::Ptr lidarCloud;
        pcl::PointCloud<pcl::PointXYZI>::Ptr lidarDenseCloud;
        float dis_threshold = 20.0;
        string dataPath;
        string outputFolder;
        string projectionFolder;
        
        string lidarFile;
        string imageEdgeSphereFile;
        string lidarDenseFile;
        string lidarMarkFile;
        string imageMarkFile;
        string rawImageFile;
        string imagePressedFile;
        string lidarFlatFile;
        string imageFlatFile;
};

#endif