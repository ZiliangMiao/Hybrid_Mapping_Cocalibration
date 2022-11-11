#include <Eigen/Core>
#include <pcl/common/common.h>

#define PI_M                (3.14159265358)
#define THREADS             (16)

#define K_INT               (10)
#define KDE_SCALE           (1)
#define SAMPLING_RADIUS     (0.01)
#define MESSAGE_EN          (1)
#define EXTRA_FILE_EN       (0)

#define MatD(a,b)           Eigen::Matrix<double, (a), (b)>
#define MatF(a,b)           Eigen::Matrix<float, (a), (b)>

typedef MatD(2,1)           Vec2D;
typedef MatF(2,1)           Vec2F;
typedef MatD(3,1)           Vec3D;
typedef MatF(3,1)           Vec3F;
typedef MatD(4,1)           Vec4D;
typedef MatF(4,1)           Vec4F;

typedef MatD(3,3)           Mat3D;
typedef MatF(3,3)           Mat3F;
typedef MatD(4,4)           Mat4D;
typedef MatF(4,4)           Mat4F;

typedef MatD(K_INT,1)       Int_D;
typedef MatF(K_INT,1)       Int_F;
typedef MatD(6,1)           Ext_D;
typedef MatF(6,1)           Ext_F;
typedef MatD(6+K_INT,1)     Param_D;
typedef MatF(6+K_INT,1)     Param_F;

typedef pcl::PointXYZI PointI;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointCloud<PointI> CloudI;
typedef pcl::PointCloud<PointRGB> CloudRGB;

typedef std::pair<int, int> Pair;