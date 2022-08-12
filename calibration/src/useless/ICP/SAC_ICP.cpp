#include <pcl/registration/ia_ransac.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <pcl/filters/passthrough.h>//　直通滤波器

#include "xyz_io.hpp"
#include "planarSeg.hpp"

using pcl::NormalEstimation;
using pcl::search::KdTree;
typedef pcl::PointXYZ PointI;
typedef pcl::PointCloud<PointI> PointCloud;


#define TEMPLATE_FILE "../data_set/templateDisinfect.xyz"
#define TARGET_FILE "../data_set/0Disinfect.xyz"

//点云可视化
void visualize_pcd(PointCloud::Ptr pcd_src,
   PointCloud::Ptr pcd_tgt,
   PointCloud::Ptr pcd_final)
{
   //int vp_1, vp_2;
   // Create a PCLVisualizer object
   pcl::visualization::PCLVisualizer viewer("registration Viewer");
   //viewer.createViewPort (0.0, 0, 0.5, 1.0, vp_1);
  // viewer.createViewPort (0.5, 0, 1.0, 1.0, vp_2);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h (pcd_src, 0, 255, 0);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h (pcd_tgt, 255, 0, 0);
   pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> final_h (pcd_final, 0, 0, 255);
   viewer.addPointCloud (pcd_src, src_h, "source cloud");
   viewer.addPointCloud (pcd_tgt, tgt_h, "tgt cloud");
   viewer.addPointCloud (pcd_final, final_h, "final cloud");
   //viewer.addCoordinateSystem(1.0);
   while (!viewer.wasStopped())
   {
       viewer.spinOnce(100);
       //boost::this_thread::sleep(boost::posix_time::microseconds(100000));
   }
}

//由旋转平移矩阵计算旋转角度
void matrix2angle (Eigen::Matrix4f &result_trans,Eigen::Vector3f &result_angle)
{
  double ax,ay,az;
  if (result_trans(2,0)==1 || result_trans(2,0)==-1)
  {
      az=0;
      double dlta;
      dlta=atan2(result_trans(0,1),result_trans(0,2));
      if (result_trans(2,0)==-1)
      {
          ay=M_PI/2;
          ax=az+dlta;
      }
      else
      {
          ay=-M_PI/2;
          ax=-az+dlta;
      }
  }
  else
  {
      ay=-asin(result_trans(2,0));
      ax=atan2(result_trans(2,1)/cos(ay),result_trans(2,2)/cos(ay));
      az=atan2(result_trans(1,0)/cos(ay),result_trans(0,0)/cos(ay));
  }
  result_angle<<ax,ay,az;
}

int
   main (int argc, char** argv)
{
   //加载点云文件
   /*
   PointCloud::Ptr cloud_src_o (new PointCloud);//原点云，待配准
   pcl::io::loadPCDFile ("bunny_rotated.pcd",*cloud_src_o);  
   PointCloud::Ptr cloud_tgt_o (new PointCloud);//目标点云
   pcl::io::loadPCDFile ("bunny.pcd",*cloud_tgt_o);
   */
   // load point cloud from XYZ file
   std::cout << "[0] Load Template Cloud." << std::endl;
   PointCloud::Ptr cloud_src_o (new PointCloud);//原点云，待配准
   PointCloud::Ptr cloud_tgt_o (new PointCloud);//目标点云
   PointCloud::Ptr cloud_tgt_O (new PointCloud);//目标点云, before range set, Z
   PointCloud::Ptr cloud_tgt_OO (new PointCloud);//目标点云, before range set,Y

   LoadXYZFile(TARGET_FILE, *cloud_tgt_OO);
   LoadXYZFile(TEMPLATE_FILE, *cloud_src_o);

   clock_t start=clock();
   //去除NAN点
   /*
   std::vector<int> indices_src; //保存去除的点的索引
   pcl::removeNaNFromPointCloud(*cloud_src_o,*cloud_src_o, indices_src);
   std::cout<<"remove *cloud_src_o nan"<<endl;
   */

   // 直通滤波，将Z轴不在（0，1.5）范围的点过滤掉，将剩余的点存储到cloud_filtered对象中
   pcl::PassThrough<pcl::PointXYZ> pass;//直通滤波对象
   pcl::PassThrough<pcl::PointXYZ> pass_Y;//直通滤波对象
   pass.setInputCloud (cloud_tgt_OO);
   pass.setFilterFieldName ("z");// Z轴
   pass.setFilterLimits (0.3, 1.7);//　范围
   pass.filter (*cloud_tgt_O); // after Z range set

   /*
   pass.setInputCloud (cloud_tgt_O);
   pass.setFilterFieldName ("y");// y轴
   pass.setFilterLimits (-0.1, 1);//　范围
   pass.filter (*cloud_tgt_o); // after Z range set
   */

   std::cerr << "Original PointCloud has: " << cloud_tgt_O->points.size () << " data points." << std::endl;
   Segmentation SEG;
   SEG.SegProcess(cloud_tgt_O, cloud_tgt_o);
   std::cerr << "PointCloud after removing planars have: " << cloud_tgt_o->points.size () << " data points." << std::endl;
   // pcl::copyPointCloud(*cloud_tgt_O, *cloud_tgt_o);


   //下采样滤波
   const float leaf1 = 0.012f;
   const float leaf2 = 0.01f; // original 0.01

   pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
   voxel_grid.setLeafSize(leaf1, leaf1, leaf1); // original 0.012
   voxel_grid.setInputCloud(cloud_src_o);
   PointCloud::Ptr cloud_src (new PointCloud);
   voxel_grid.filter(*cloud_src);
   std::cout<<"down size *cloud_src_o from "<<cloud_src_o->size()<<"to"<<cloud_src->size()<<endl;
   pcl::io::savePCDFileASCII("bunny_src_down.pcd",*cloud_src);

   //计算表面法线
   pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_src;
   ne_src.setInputCloud(cloud_src);
   pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZ>());
   ne_src.setSearchMethod(tree_src);
   pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
   ne_src.setRadiusSearch(0.02);
   ne_src.compute(*cloud_src_normals);
   
   /*
   std::vector<int> indices_tgt;
   pcl::removeNaNFromPointCloud(*cloud_tgt_o,*cloud_tgt_o, indices_tgt);
   std::cout<<"remove *cloud_tgt_o nan"<<endl;
   */

   pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_2;
   voxel_grid_2.setLeafSize(leaf2, leaf2, leaf2);
   voxel_grid_2.setInputCloud(cloud_tgt_o);
   PointCloud::Ptr cloud_tgt (new PointCloud);
   voxel_grid_2.filter(*cloud_tgt);
   std::cout<<"down size *cloud_tgt_o.pcd from "<<cloud_tgt_o->size()<<"to"<<cloud_tgt->size()<<endl;
   pcl::io::savePCDFileASCII("bunny_tgt_down.pcd",*cloud_tgt);

   pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_tgt;
   ne_tgt.setInputCloud(cloud_tgt);
   pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZ>());
   ne_tgt.setSearchMethod(tree_tgt);
   pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud< pcl::Normal>);
   //ne_tgt.setKSearch(20);
   ne_tgt.setRadiusSearch(0.02);
   ne_tgt.compute(*cloud_tgt_normals);

   //计算FPFH
   pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_src;
   fpfh_src.setInputCloud(cloud_src);
   fpfh_src.setInputNormals(cloud_src_normals);
   pcl::search::KdTree<PointI>::Ptr tree_src_fpfh (new pcl::search::KdTree<PointI>);
   fpfh_src.setSearchMethod(tree_src_fpfh);
   pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
   fpfh_src.setRadiusSearch(0.05);
   fpfh_src.compute(*fpfhs_src);
   std::cout<<"compute *cloud_src fpfh"<<endl;

   pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_tgt;
   fpfh_tgt.setInputCloud(cloud_tgt);
   fpfh_tgt.setInputNormals(cloud_tgt_normals);
   pcl::search::KdTree<PointI>::Ptr tree_tgt_fpfh (new pcl::search::KdTree<PointI>);
   fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
   pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
   fpfh_tgt.setRadiusSearch(0.05);
   fpfh_tgt.compute(*fpfhs_tgt);
   std::cout<<"compute *cloud_tgt fpfh"<<endl;

   //SAC配准
   pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
   scia.setInputSource(cloud_src);
   scia.setInputTarget(cloud_tgt);
   scia.setSourceFeatures(fpfhs_src);
   scia.setTargetFeatures(fpfhs_tgt);
   //scia.setMinSampleDistance(1);
   //scia.setNumberOfSamples(2);
   //scia.setCorrespondenceRandomness(20);
   PointCloud::Ptr sac_result (new PointCloud);
   scia.align(*sac_result);
   std::cout  <<"sac has converged:"<<scia.hasConverged()<<"  score: "<<scia.getFitnessScore()<<endl;
   Eigen::Matrix4f sac_trans;
   sac_trans=scia.getFinalTransformation();
   std::cout<<sac_trans<<endl;
   pcl::io::savePCDFileASCII("bunny_transformed_sac.pcd",*sac_result);
   clock_t sac_time=clock();

   //icp配准
   PointCloud::Ptr icp_result (new PointCloud);
   pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
   icp.setInputSource(cloud_src);
   icp.setInputTarget(cloud_tgt_o);
   //Set the max correspondence distance to 4cm (e.g., correspondences with higher distances will be ignored)
   icp.setMaxCorrespondenceDistance (0.04);
   // 最大迭代次数
   icp.setMaximumIterations (50);
   // 两次变化矩阵之间的差值
   icp.setTransformationEpsilon (1e-10);
   // 均方误差
   icp.setEuclideanFitnessEpsilon (0.2);
   icp.align(*icp_result,sac_trans);

   clock_t end=clock();
   cout<<"total time: "<<(double)(end-start)/(double)CLOCKS_PER_SEC<<" s"<<endl;
   //我把计算法线和点特征直方图的时间也算在SAC里面了
   cout<<"sac time: "<<(double)(sac_time-start)/(double)CLOCKS_PER_SEC<<" s"<<endl;
   cout<<"icp time: "<<(double)(end-sac_time)/(double)CLOCKS_PER_SEC<<" s"<<endl;

   std::cout << "ICP has converged:" << icp.hasConverged()
       << " score: " << icp.getFitnessScore() << std::endl;
   Eigen::Matrix4f icp_trans;
   icp_trans=icp.getFinalTransformation();
   //cout<<"ransformationProbability"<<icp.getTransformationProbability()<<endl;
   std::cout<<icp_trans<<endl;
   //使用创建的变换对未过滤的输入点云进行变换
   pcl::transformPointCloud(*cloud_src_o, *icp_result, icp_trans);
   //保存转换的输入点云
   pcl::io::savePCDFileASCII("bunny_transformed_sac_ndt.pcd", *icp_result);

   //计算误差
   Eigen::Vector3f ANGLE_origin;
   ANGLE_origin<<0,0,M_PI/5;
   double error_x,error_y,error_z;
   Eigen::Vector3f ANGLE_result;
   matrix2angle(icp_trans,ANGLE_result);
   error_x=fabs(ANGLE_result(0))-fabs(ANGLE_origin(0));
   error_y=fabs(ANGLE_result(1))-fabs(ANGLE_origin(1));
   error_z=fabs(ANGLE_result(2))-fabs(ANGLE_origin(2));
   cout<<"original angle in x y z:\n"<<ANGLE_origin<<endl;
   cout<<"error in aixs_x: "<<error_x<<"  error in aixs_y: "<<error_y<<"  error in aixs_z: "<<error_z<<endl;

   //可视化
   visualize_pcd(cloud_src_o,cloud_tgt_o,icp_result);
   return (0);
}