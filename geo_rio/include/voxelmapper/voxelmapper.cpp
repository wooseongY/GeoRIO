#include "voxelmapper.h"

// TODO : use both gaussian + plane 
void VoxelOctoTree::initPlane(const Eigen::aligned_deque<PointData> &points, VoxelPlane *plane)
{
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points)
  {
    Eigen::Vector3d point_w = Eigen::Vector3d(pv.pt_w.x, pv.pt_w.y, pv.pt_w.z);
    plane->covariance_ += point_w * point_w.transpose();
    plane->center_ += point_w;
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  // evalsReal /= evalsReal.maxCoeff();
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  if (evalsReal(evalsMin) < planer_threshold_)
  {
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_;

    for (int i = 0; i < points.size(); i++)
    {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      Eigen::Vector3d point_w = Eigen::Vector3d(points[i].pt_w.x, points[i].pt_w.y, points[i].pt_w.z);
      for (int m = 0; m < 3; m++)
      {
        if (m != (int)evalsMin)
        {
          Eigen::Matrix<double, 1, 3> F_m =
              (point_w - plane->center_).transpose() / ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        }
        else
        {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].cov * J.transpose(); // var: world frame covariance
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_)
    {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  }
  else
  {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

void VoxelOctoTree::initOctoTree()
{
  if (temp_points_.size() > points_size_threshold_)
  {
    initPlane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true)
    {
      octo_state_ = 0;
      // new added
      if (temp_points_.size() > max_points_num_)
      {
        update_enable_ = false;
        Eigen::aligned_deque<PointData>().swap(temp_points_);
        new_points_ = 0;
      }
    }
    else
    {
      octo_state_ = 1;
      cutOctoTree();
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::cutOctoTree()
{
  if (layer_ >= max_layer_)
  {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].pt_w.x > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].pt_w.y > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].pt_w.z > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        initPlane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;
            Eigen::aligned_deque<PointData>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cutOctoTree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::updateOctoTree(const PointData &pv)
{
  if (!init_octo_)
  {
    new_points_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > points_size_threshold_) { initOctoTree(); }
  }
  else
  {
    if (plane_ptr_->is_plane_)
    {
      if (update_enable_)
      {
        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_)
        {
          initPlane(temp_points_, plane_ptr_);
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_)
        {
          update_enable_ = false;
          Eigen::aligned_deque<PointData>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    }
    else
    {
      if (layer_ < max_layer_)
      {
        int xyz[3] = {0, 0, 0};
        if (pv.pt_w.x > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.pt_w.y > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.pt_w.z > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->updateOctoTree(pv); }
        else
        {
          leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->updateOctoTree(pv);
        }
      }
      else
      {
        if (update_enable_)
        {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_)
          {
            initPlane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_)
          {
            update_enable_ = false;
            Eigen::aligned_deque<PointData>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

// currently not used
VoxelOctoTree *VoxelOctoTree::find_correspondence(Eigen::Vector3d pw)
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspondence(pw) : this;
}

VoxelOctoTree *VoxelOctoTree::insertPoint(const PointData &pv)
{
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_)))
  {
    new_points_++;
    temp_points_.push_back(pv);
    return this;
  }

  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_))
  {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.pt_w.x > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.pt_w.y > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.pt_w.z > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) { return leaves_[leafnum]->insertPoint(pv); }
    else
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      return leaves_[leafnum]->insertPoint(pv);
    }
  }
  return nullptr;
}

void VoxelMapManager::calcBodyCov(Eigen::Vector3d &pb, Eigen::Matrix3d &cov) 
{
  if (pb[2] == 0) pb[2] = 0.000001; // 0.0001
  float range = std::sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = config_setting_.range_acc_ * config_setting_.range_acc_;

  Eigen::Matrix2d direction_var;
  // direction_var << pow(sin(DEG2RAD(azi_unc)), 2), 0, 0, pow(sin(DEG2RAD(elev_unc)), 2); // degree
  direction_var << pow(DEG2RAD(config_setting_.azi_acc_), 2), 0, 0, pow(DEG2RAD(config_setting_.elev_acc_ ), 2); // radian
  
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d azimuth_dir(-pb[1], pb[0], 0.0);
  if (azimuth_dir.norm() < 1e-9) {
    azimuth_dir = Eigen::Vector3d(1.0, 0.0, 0.0);
  }
  azimuth_dir.normalize();
  Eigen::Vector3d elevation_dir = direction.cross(azimuth_dir);
  elevation_dir.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N.col(0) = azimuth_dir;
  N.col(1) = elevation_dir;
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();

  // cov = 0.05 * cov;
}


void VoxelMapManager::buildVoxelMap(int64_t start_t_ns, const SplineState* spl) //with spline?
{
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Building voxel map...");
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.plane_thresh_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;

  // std::vector<PointData> pt_data_buff_;
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "pt_data_buff_ size: %zu", pt_data_buff_.size());

  // for (size_t i = 0; i < pt_data_buff_.size(); i++)
  // {
  //   PointData& pv = pt_data_buff_[i];

  //   Eigen::Matrix3d cov;
  //   calcBodyCov(pv.pt_l, cov);

  //   Eigen::Matrix3d point_crossmat;
  //   point_crossmat << 0, -pv.pt_l(2), pv.pt_l(1),
  //                     pv.pt_l(2), 0, -pv.pt_l(0),
  //                     -pv.pt_l(1), pv.pt_l(0), 0;
    
  //   Eigen::Quaterniond q_itp;

  //   spl->itpQuaternion(start_t_ns, &q_itp);
  //   Eigen::Vector3d p_itp = spl->itpPosition(start_t_ns);
  //   // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "q_itp: %f, %f, %f, %f", q_itp.x(), q_itp.y(), q_itp.z(), q_itp.w());
  //   // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "p_itp: %f, %f, %f", p_itp.x(), p_itp.y(), p_itp.z());

  //   // TODO: use state uncertainty from spline
  //   // cov = (state_.rot_end * extR_) * cov * (state_.rot_end * extR_).transpose() +
  //   //       (-point_crossmat) * state_.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + state_.cov.block<3, 3>(3, 3);
  //   pv.cov = cov;
  //   // pt_data_buff_.push_back(pv);
  // }

  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Converted...");

  uint plsize = pt_data_buff_.size();
  for (uint i = 0; i < plsize; i++)
  {
    const PointData p_v = pt_data_buff_[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.pt_w.data[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
    }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
    }
  }
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter)
  {
    iter->second->initOctoTree();
  }
}

void VoxelMapManager::updateVoxelMap(const Eigen::aligned_deque<PointData> &input_points)
{
  RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Updating voxel map...");
  float voxel_size = config_setting_.max_voxel_size_;
  float planar_threshold = config_setting_.plane_thresh_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const PointData p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_v.pt_w.data[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end()) { voxel_map_[position]->updateOctoTree(p_v); }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planar_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->updateOctoTree(p_v);
    }
  }
}

void VoxelMapManager::findCorrespondVoxel(int &num_tot_eff, Eigen::aligned_deque<PointData> &pv_list)
{
  RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Finding corresponding voxels...");
  int no_voxel_num = 0;
  // need to compute pointdata. normvec, if valid, plane coeff (d)
  
  int max_layer = config_setting_.max_layer_;
  double voxel_size = config_setting_.max_voxel_size_;
  double sigma_num = config_setting_.sigma_num_;
  std::mutex mylock;
  
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
  }

  // #ifdef MP_EN
  //   omp_set_num_threads(MP_PROC_NUM);
  //   #pragma omp parallel for
  // #endif
  for (int i = 0; i < index.size(); i++)
  {
    RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "-------Finding correspondence for point %d-------", i);
    PointData &pv = pv_list[i];
    pv.if_valid = false;
    // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "for pointdata %f, %f, %f", pv.pt_w.x, pv.pt_w.y, pv.pt_w.z);

    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pv.pt_w.data[j] / voxel_size;
      if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
    }
    VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      VoxelOctoTree *current_octo = iter->second;
      bool is_success = false;
      
      double prob = 0;
      findSinglePtCorrespondVoxel(pv, current_octo, 0, is_success, prob);
      if (!is_success)
      {
        VOXEL_LOCATION near_position = position;
        if (loc_xyz[0] > (current_octo->voxel_center_[0] + current_octo->quater_length_)) { near_position.x = near_position.x + 1; }
        else if (loc_xyz[0] < (current_octo->voxel_center_[0] - current_octo->quater_length_)) { near_position.x = near_position.x - 1; }
        if (loc_xyz[1] > (current_octo->voxel_center_[1] + current_octo->quater_length_)) { near_position.y = near_position.y + 1; }
        else if (loc_xyz[1] < (current_octo->voxel_center_[1] - current_octo->quater_length_)) { near_position.y = near_position.y - 1; }
        if (loc_xyz[2] > (current_octo->voxel_center_[2] + current_octo->quater_length_)) { near_position.z = near_position.z + 1; }
        else if (loc_xyz[2] < (current_octo->voxel_center_[2] - current_octo->quater_length_)) { near_position.z = near_position.z - 1; }
        auto iter_near = voxel_map_.find(near_position);
        if (iter_near != voxel_map_.end()) {
          findSinglePtCorrespondVoxel(pv, iter_near->second, 0, is_success, prob);
        }
      }
      RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "is success: %d", is_success);

      // if (!is_success)
      // {
      //   RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Reached max layer %d, plane is_plane: %d",
      //               max_layer, current_octo->plane_ptr_->is_plane_);
      //   RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "current_octo->plane_ptr_->points_size_: %d",
      //               current_octo->plane_ptr_->points_size_);
      //   Eigen::EigenSolver<Eigen::Matrix3d> es(current_octo->plane_ptr_->covariance_);
      //   Eigen::Matrix3cd evecs = es.eigenvectors();
      //   Eigen::Vector3cd evals = es.eigenvalues();
      //   Eigen::Vector3d evalsReal;
      //   evalsReal = evals.real();
      //   Eigen::Matrix3f::Index evalsMin, evalsMax;
      //   evalsReal.rowwise().sum().minCoeff(&evalsMin);
      //   evalsReal.rowwise().sum().maxCoeff(&evalsMax);
      //   int evalsMid = 3 - evalsMin - evalsMax;
      //   Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
      //   // Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
      //   // Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
      //   pv.vox_center = current_octo->plane_ptr_->center_;

      //   pv.normvec << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
      //   pv.dist = -(pv.normvec(0) * current_octo->plane_ptr_->center_(0) + pv.normvec(1) * current_octo->plane_ptr_->center_(1) + pv.normvec(2) * current_octo->plane_ptr_->center_(2));
      //   pv.if_valid = true;
      //   is_success = true;
      // }

      if(is_success){
        // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Found correspondence in voxel location (%f, %f, %f) with size %f at layer %d",
        //             current_octo->voxel_center_[0], current_octo->voxel_center_[1], current_octo->voxel_center_[2], current_octo->quater_length_ * 2.0, current_octo->layer_);
        num_tot_eff++;
      }
    }
    else{
      no_voxel_num++;
      // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "No voxel found for pointdata %f, %f, %f", pv.pt_w.x, pv.pt_w.y, pv.pt_w.z);
    }
  }
  RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "total number of points: %zu", pv_list.size());
  RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "number of points with no voxel correspondence: %d", no_voxel_num);
  RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "number of points with voxel correspondence: %d", num_tot_eff);
  RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "voxel is not plane: %ld", pv_list.size() - num_tot_eff - no_voxel_num);
}

void VoxelMapManager::findSinglePtCorrespondVoxel(PointData &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_success,
                                            double &prob)
{
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Finding single point correspondence in voxel map...");
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "for pointdata %f, %f, %f", pv.pt_w.x, pv.pt_w.y, pv.pt_w.z);
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Finding correspondence in voxel location (%f, %f, %f) with size %f at layer %d",
  //             current_octo->voxel_center_[0], current_octo->voxel_center_[1], current_octo->voxel_center_[2], current_octo->quater_length_ * 2.0, current_layer);
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "current_octo->plane_ptr_->is_plane_: %d", current_octo->plane_ptr_->is_plane_);
  int max_layer = config_setting_.max_layer_;
  double sigma_num = config_setting_.sigma_num_;

  double radius_k = 3;
  Eigen::Vector3d p_w = Eigen::Vector3d(pv.pt_w.x, pv.pt_w.y, pv.pt_w.z);
  if (current_octo->plane_ptr_->is_plane_)
  {
    // pv.vox_center = current_octo->plane_ptr_->center_;

    VoxelPlane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center_;
    float dis_to_plane = fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_);
    float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) + (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                          (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);
    
    if (range_dis <= radius_k * plane.radius_)
    {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
      J_nq.block<1, 3>(0, 3) = -plane.normal_;
      // compute plane covariance
      double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
      sigma_l += plane.normal_.transpose() * pv.cov * plane.normal_;
      // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "dis_to_plane: %f, range_dis: %f", dis_to_plane, range_dis);
      // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "sigma_l: %f", sigma_l);
      // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "plane radius: %f", plane.radius_);
      // if (dis_to_plane < sigma_num * sqrt(sigma_l))
      // {
        is_success = true;
        // double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        // if (this_prob > prob)
        // {
          // prob = this_prob;
          pv.normvec = plane.normal_;
          pv.dist = plane.d_;
          pv.if_valid = true;
          // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "for pointdata %f, %f, %f", pv.pt_w.x, pv.pt_w.y, pv.pt_w.z);
          // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Found correspondence in voxel location (%f, %f, %f) with size %f at layer %d with prob %f",
          //             current_octo->voxel_center_[0], current_octo->voxel_center_[1], current_octo->voxel_center_[2], current_octo->quater_length_ * 2.0, current_layer, prob);
        // }
        // return;
      // }
      // else
      // {
      //   // is_success = false;
      //   return;
      // }
    }
    else
    {
      // is_success = false;
      return;
    }
  }
  else
  {
    if (current_layer < max_layer)
    {
      for (size_t leafnum = 0; leafnum < 8; leafnum++)
      {
        if (current_octo->leaves_[leafnum] != nullptr)
        {
          VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
          findSinglePtCorrespondVoxel(pv, leaf_octo, current_layer + 1, is_success, prob);
        }
      }
      return;
    }
    else {
      return;
    }
  }
}

void VoxelMapManager::mapSliding(int64_t time_ns, SplineState* spline)
{
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapManager"), "Sliding Window Voxel Map...");
  position_last_ = spline->itpPosition(time_ns);
  if((position_last_ - last_slide_position).norm() < config_setting_.sliding_thresh)
  {
    // std::cout<<RED<<"[DEBUG]: Last sliding length "<<(position_last_ - last_slide_position).norm()<<RESET<<"\n";
    return;
  }

  // get global id now
  last_slide_position = position_last_;
  double t_sliding_start = omp_get_wtime();
  float loc_xyz[3];
  for (int j = 0; j < 3; j++)
  {
    loc_xyz[j] = position_last_[j] / config_setting_.max_voxel_size_;
    if (loc_xyz[j] < 0) { loc_xyz[j] -= 1.0; }
  }
  // VOXEL_LOCATION position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);//discrete global
  clearMemOutOfMap((int64_t)loc_xyz[0] + config_setting_.half_map_size, (int64_t)loc_xyz[0] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[1] + config_setting_.half_map_size, (int64_t)loc_xyz[1] - config_setting_.half_map_size,
                    (int64_t)loc_xyz[2] + config_setting_.half_map_size, (int64_t)loc_xyz[2] - config_setting_.half_map_size);
  double t_sliding_end = omp_get_wtime();
  // std::cout<<RED<<"[DEBUG]: Map sliding using "<<t_sliding_end - t_sliding_start<<" secs"<<RESET<<"\n";
  return;
}

void VoxelMapManager::clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min )
{
  int delete_voxel_cout = 0;
  // double delete_time = 0;
  // double last_delete_time = 0;
  for (auto it = voxel_map_.begin(); it != voxel_map_.end(); )
  {
    const VOXEL_LOCATION& loc = it->first;
    bool should_remove = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min || loc.z > z_max || loc.z < z_min;
    if (should_remove){
      // last_delete_time = omp_get_wtime();
      delete it->second;
      it = voxel_map_.erase(it);
      // delete_time += omp_get_wtime() - last_delete_time;
      delete_voxel_cout++;
    } else {
      ++it;
    }
  }
  // std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" root voxels"<<RESET<<"\n";
  // std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" voxels using "<<delete_time<<" s"<<RESET<<"\n";
}

void VoxelMapManager::pubVoxelMap(){
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapper"), "Publishing voxel map...");
  double max_trace = 0.25;
  double pow_num = 0.2;
  rclcpp::Rate rate(500);
  float use_alpha = 0.8;
  visualization_msgs::msg::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<VoxelPlane> pub_plane_list;
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++)
  {
    GetUpdatePlane(iter->second, config_setting_.max_layer_, pub_plane_list);
  }
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapper"), "Get %zu planes to publish.", pub_plane_list.size());
  for (size_t i = 0; i < pub_plane_list.size(); i++)
  {
    Eigen::Vector3d plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0, 0).diagonal();
    // double trace = plane_cov.sum();
    // if (trace >= max_trace) { trace = max_trace; }
    // trace = trace * (1.0 / max_trace);
    // trace = pow(trace, pow_num);
    uint8_t r, g, b;
    // mapJet(trace, 0, 1, r, g, b);
    r = 128;
    g = 128;
    b = 128;
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane_) { alpha = use_alpha; }
    else { alpha = 0; }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapper"), "Publishing %zu planes.", voxel_plane.markers.size());
  voxel_map_pub_->publish(voxel_plane);
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapper"), "Voxel map published.");
  rate.sleep();
}

void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list)
{
  if (current_octo->layer_ > pub_max_voxel_layer) { return; }
  if (current_octo->plane_ptr_->is_update_) { plane_list.push_back(*current_octo->plane_ptr_); }
  if (current_octo->layer_ < current_octo->max_layer_)
  {
    if (!current_octo->plane_ptr_->is_plane_)
    {
      for (size_t i = 0; i < 8; i++)
      {
        if (current_octo->leaves_[i] != nullptr) { GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list); }
      }
    }
  }
  return;
}


void VoxelMapManager::pubSinglePlane(visualization_msgs::msg::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane,
                                     const float alpha, const Eigen::Vector3d rgb)
{
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapper"), "Publishing plane %d with alpha %f", single_plane.id_, alpha);
  visualization_msgs::msg::Marker plane;
  plane.header.frame_id = "odom";
  plane.header.stamp = rclcpp::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id_;
  plane.type = visualization_msgs::msg::Marker::SPHERE;
  plane.action = visualization_msgs::msg::Marker::ADD;
  plane.pose.position.x = single_plane.center_[0];
  plane.pose.position.y = single_plane.center_[1];
  plane.pose.position.z = single_plane.center_[2];
  geometry_msgs::msg::Quaternion q;
  CalcVectQuation(single_plane.x_normal_, single_plane.y_normal_, single_plane.normal_, q);
  plane.pose.orientation = q;
  // RCLCPP_INFO(rclcpp::get_logger("VoxelMapper"), "Plane pose: %f, %f, %f, %f, %f, %f, %f", plane.pose.position.x, plane.pose.position.y,
  //          plane.pose.position.z, q.x, q.y, q.z, q.w);
  
  if (std::isnan(single_plane.max_eigen_value_) || std::isnan(single_plane.mid_eigen_value_) || std::isnan(single_plane.min_eigen_value_))
  {
    // RCLCPP_ERROR(rclcpp::get_logger("VoxelMapper"), "Plane eigen values are NaN, skipping plane %d", single_plane.id_);
    return;
  }
  if(single_plane.max_eigen_value_ <= 0 || single_plane.mid_eigen_value_ <= 0 || single_plane.min_eigen_value_ <= 0)
  {
    // RCLCPP_ERROR(rclcpp::get_logger("VoxelMapper"), "Plane eigen values are negative, skipping plane %d", single_plane.id_);
    return;
  }
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value_);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value_);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value_);
  
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = rclcpp::Duration(0, 0);
  plane_pub.markers.push_back(plane);
}

void VoxelMapManager::CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                                      geometry_msgs::msg::Quaternion &q)
{
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}