#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <visualization_msgs/msg/marker_array.hpp>

#include "utils/common_utils.h"
#include "SplineState.h"

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

static int voxel_plane_id = 0;

struct VoxelConfig
{
    double max_voxel_size_;
    int max_layer_;
    int max_iterations_;
    std::vector<int> layer_init_num_;
    int max_points_num_;
    double plane_thresh_;
    double range_acc_;
    double azi_acc_;
    double elev_acc_;
    double sigma_num_;
    bool pub_voxelmap_en_;

    // config of local map sliding
    bool map_sliding_en;
    int half_map_size;
    double sliding_thresh;

    VoxelConfig() = default;

    VoxelConfig(rclcpp::Node::SharedPtr& nh) {
        // (23): per-detection radar noise
        range_acc_ = CommonUtils::readParam<double>(nh, "radar_noise.range_std", 0.2);
        azi_acc_ = CommonUtils::readParam<double>(nh, "radar_noise.azimuth_std", 0.5);
        elev_acc_ = CommonUtils::readParam<double>(nh, "radar_noise.elevation_std", 0.5);
        max_layer_ = 3;
        max_voxel_size_ = 4.0;
        layer_init_num_ = std::vector<int>(7, 5);
        max_points_num_ = 1000;
        plane_thresh_ = 0.1;
        map_sliding_en = false;
        half_map_size = 500;
        sliding_thresh = 8.0;
        pub_voxelmap_en_ = false;
    }
};

typedef struct VoxelPlane
{
    Eigen::Vector3d center_;
    Eigen::Vector3d normal_;
    Eigen::Vector3d y_normal_;
    Eigen::Vector3d x_normal_;
    Eigen::Matrix3d covariance_;
    Eigen::Matrix<double, 6, 6> plane_var_;
    float radius_ = 0;
    float min_eigen_value_ = 1;
    float mid_eigen_value_ = 1;
    float max_eigen_value_ = 1;
    float d_ = 0;
    int points_size_ = 0;
    bool is_plane_ = false;
    bool is_init_ = false;
    int id_ = 0;
    bool is_update_ = false;
    VoxelPlane()
    {
        plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
        covariance_ = Eigen::Matrix3d::Zero();
        center_ = Eigen::Vector3d::Zero();
        normal_ = Eigen::Vector3d::Zero();
    }
} VoxelPlane;

// TODO: use gaussian?
typedef struct VoxelGaussian
{
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  float d_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
  bool is_init_ = false;
  int id_ = 0;
  bool is_update_ = false;
  VoxelGaussian()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelGaussian;



class VOXEL_LOCATION
{
public:
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  size_t operator()(const VOXEL_LOCATION &loc) const
  {
    return (loc.x * VOXELMAP_HASH_P + loc.y * VOXELMAP_HASH_P + loc.z * VOXELMAP_HASH_P) % VOXELMAP_MAX_N;
  }
};
} // namespace std


class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  Eigen::aligned_deque<PointData> temp_points_;
  VoxelPlane *plane_ptr_;
  VoxelGaussian *gaussian_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  VoxelOctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_init_num_;
  float quater_length_;
  float planer_threshold_;
  int points_size_threshold_;
  int update_size_threshold_;
  int max_points_num_;
  int max_layer_;
  int new_points_;
  bool init_octo_;
  bool update_enable_;

  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new VoxelPlane;
    gaussian_ptr_ = new VoxelGaussian;
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  void initPlane(const Eigen::aligned_deque<PointData> &points, VoxelPlane *plane);
  void initOctoTree();
  void cutOctoTree();
  void updateOctoTree(const PointData &pv);

  VoxelOctoTree *find_correspondence(Eigen::Vector3d pw);
  VoxelOctoTree *insertPoint(const PointData &pv);
};

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelConfig config_setting_;
  int current_frame_id_ = 0;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_map_pub_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;

  // pcl::PointCloud<pcl::PointXYZINormal>::Ptr feats_undistort_;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr feats_down_body_;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr feats_down_world_;

  Eigen::aligned_deque<PointData> pt_data_buff_;

//   Eigen::Matrix3d extR_;
//   Eigen::Vector3d extT_;
  float build_residual_time, ekf_time;
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;
  int scan_count = 0;
//   StatesGroup state_;
  Eigen::Vector3d position_last_;

  Eigen::Vector3d last_slide_position = {0,0,0};

  int feats_down_size_;
  int effct_feat_num_;
  std::vector<Eigen::Matrix3d> cross_mat_list_;
  std::vector<Eigen::Matrix3d> body_cov_list_;
  std::vector<PointData> pv_list_;

  VoxelMapManager(VoxelConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map)
  {
    current_frame_id_ = 0;
    // feats_undistort_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    feats_down_body_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    feats_down_world_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
    pt_data_buff_.clear();
  };

  void calcBodyCov(Eigen::Vector3d &pb, Eigen::Matrix3d &cov) ;
//   void StateEstimation(StatesGroup &state_propagat);
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

  void buildVoxelMap(int64_t start_t_ns, const SplineState *spline);
//   V3F RGBFromVoxel(const Eigen::Vector3d &input_point);

  void updateVoxelMap(const Eigen::aligned_deque<PointData> &input_points);

  void findCorrespondVoxel(int &num_tot_eff, Eigen::aligned_deque<PointData> &pv_list);

  void findSinglePtCorrespondVoxel(PointData &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob);

  void mapSliding(int64_t time_ns, SplineState* spline);
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );

  void pubVoxelMap();

private:
  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);

  void pubSinglePlane(visualization_msgs::msg::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane, const float alpha,
                      const Eigen::Vector3d rgb);
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::msg::Quaternion &q);

//   void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;
