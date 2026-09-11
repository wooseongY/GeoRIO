#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <thread>
#include <iostream>
#include <queue>
#include <string>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/int64.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <rclcpp/service.hpp>
#include <std_srvs/srv/empty.hpp>
#include "livox_ros_driver/msg/custom_msg.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "livox_interfaces/msg/custom_msg.hpp"
#include "estimate_msgs/msg/calib.hpp"
#include "estimate_msgs/msg/estimate.hpp"
#include "SplineState.h"
#include <random>
#include <cmath>

struct RadarRansacConfig {
    int sample_size = 5;
    double outlier_prob = 0.05;
    double success_prob = 0.995;
    double inlier_threshold = 0.1;
    std::size_t min_static_points = 30;
    std::string mode = "3d";
};

static RadarRansacConfig defaultRadarRansacConfig(const std::string& lidar_type)
{
    RadarRansacConfig config;
    if (lidar_type == "HerculesContinental") {
        config.outlier_prob = 0.1;
    }
    if (lidar_type == "HkustContinental") {
        config.inlier_threshold = 0.2;
    }
    return config;
}

static RadarRansacConfig loadRadarRansacConfig(rclcpp::Node::SharedPtr& nh,
                                               const std::string& prefix,
                                               const std::string& lidar_type)
{
    RadarRansacConfig config = defaultRadarRansacConfig(lidar_type);
    const std::string ransac_prefix = prefix + "ransac.";
    config.sample_size = CommonUtils::readParam<int>(nh, ransac_prefix + "sample_size", config.sample_size);
    config.outlier_prob = CommonUtils::readParam<double>(nh, ransac_prefix + "outlier_prob", config.outlier_prob);
    config.success_prob = CommonUtils::readParam<double>(nh, ransac_prefix + "success_prob", config.success_prob);
    config.inlier_threshold = CommonUtils::readParam<double>(nh, ransac_prefix + "inlier_threshold", config.inlier_threshold);
    config.min_static_points = static_cast<std::size_t>(
        CommonUtils::readParam<int>(nh, ransac_prefix + "min_static_points", static_cast<int>(config.min_static_points)));
    config.mode = CommonUtils::readParam<std::string>(nh, ransac_prefix + "mode", config.mode);
    return config;
}

static int computeRadarRansacIterations(const RadarRansacConfig& config)
{
    if (config.sample_size <= 0) {
        return 1;
    }

    const double numerator = std::log(1.0 - config.success_prob);
    const double denominator = std::log(1.0 - std::pow(1.0 - config.outlier_prob, config.sample_size));
    if (!std::isfinite(numerator) || !std::isfinite(denominator) || std::abs(denominator) < 1.0e-12) {
        return 1;
    }

    return std::max(1, static_cast<int>(numerator / denominator));
}

static double computeRadarNormalizationDenominator(const pcl::PointXYZINormal& radar_pt,
                                                   const RadarRansacConfig& config)
{
    const bool use_3d_ransac = config.mode != "2d" && config.mode != "2D";
    double range_sq = radar_pt.x * radar_pt.x + radar_pt.y * radar_pt.y;
    if (use_3d_ransac) {
        range_sq += radar_pt.z * radar_pt.z;
    }
    const double range_norm = std::sqrt(range_sq);
    return range_norm > 1.0e-9 ? range_norm : 1.0e-9;
}

template<typename PointType>
class MappingBase
{
  public:

    std::mutex mtx;    
    LidarConfig lidar;
    RadarRansacConfig radar_ransac_config;
    MappingBase(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config)
        : lidar(lidar_config), radar_ransac_config(ransac_config)
    {
        pub_global_map = nh->create_publisher<sensor_msgs::msg::PointCloud2>("global_map", 2);
        ds_filter_each_scan.setLeafSize(0.2, 0.2, 0.2);
        pc_last.reset(new typename pcl::PointCloud<PointType>());
        pc_last_ds.reset(new typename pcl::PointCloud<PointType>());
        static_scan.reset(new typename pcl::PointCloud<PointType>());
        pc.reset(new typename pcl::PointCloud<PointType>());   
    }

    void processScan(SplineState* spl, const int64_t spl_window_st_ns)
    {
        int64_t t_end_ns = 0;
        rclcpp::Rate rate(100);
        while (!pc_L_buff.empty()) {
            t_end_ns = pc_L_buff.front().header.stamp + int64_t (pc_L_buff.front().points.back().intensity * float(1e6));
            mtx.lock();
            if (t_end_ns < spl->minTimeNs()) {
                pc_L_buff.pop_front();
                mtx.unlock(); 
            } else if (t_end_ns <= spl->maxTimeNs()) {
                transformCloud(pc_L_buff.front(), spl, pc);
                pc_L_buff.pop_front();
                mtx.unlock();                
                publishMap(pc, pub_global_map);
            } else {
                mtx.unlock(); 
                rate.sleep();
                break;
            }
        }
    }

    void publishMap(const typename pcl::PointCloud<PointType>::Ptr& pcs,
                         const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher) const
    {
        sensor_msgs::msg::PointCloud2 msgs;
        pcl::toROSMsg(*pcs, msgs);
        msgs.header.frame_id = odom_id;
        publisher->publish(msgs);
    }

  private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_global_map;

    PointType transformPoint(int64_t time_ns, const SplineState* spl, const PointType& pt_in) const
    {     
        Eigen::Quaterniond q_itp;
        Eigen::Vector3d t_itp;
        spl->itpQuaternion(time_ns, &q_itp);
        t_itp = spl->itpPosition(time_ns);
        Eigen::Vector3d p_body(pt_in.x, pt_in.y, pt_in.z);
        Eigen::Vector3d p_imu(lidar.q_bl * p_body + lidar.t_bl);
        Eigen::Vector3d p_global(q_itp * p_imu + t_itp);
        PointType point_world;
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = pt_in.intensity;
        point_world.curvature = pt_in.curvature;
        return point_world;
    }    

    void transformCloud(const typename pcl::PointCloud<PointType>& pc_in, SplineState* spl,
                       typename pcl::PointCloud<PointType>::Ptr pc_out) const
    {
        int64_t time_begin = rclcpp::Time(pc_in.header.stamp).nanoseconds();
        pc->clear();
        pc_out->points.resize(pc_in.size());
        for (size_t i = 0; i < pc_in.size(); i++) {
            const PointType& pt = pc_in.points[i];
            int64_t t_ns = int64_t(pt.intensity * float(1e6)) + time_begin;
            if (t_ns >= spl->minTimeNs() && t_ns <= spl->maxTimeNs()) {
                pc_out->points[i] = transformPoint(t_ns, spl, pt);
            }
        }
    }    

  protected:
    Eigen::aligned_deque<typename pcl::PointCloud<PointType>> pc_L_buff;
    typename pcl::PointCloud<PointType>::Ptr pc_last;
    typename pcl::PointCloud<PointType>::Ptr pc_last_ds;
    typename pcl::PointCloud<PointType>::Ptr static_scan;
    pcl::VoxelGrid<pcl::PointXYZINormal> ds_filter_each_scan;    
    const std::string frame_id = "base_link";
    const std::string odom_id = "odom";
    typename pcl::PointCloud<PointType>::Ptr pc;

};


class OusterBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  OusterBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_ouster = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&OusterBuff::ousterLidarCallback, this, std::placeholders::_1));         
        double lidar_time_offset = CommonUtils::readParam<double>(nh, "lidar_time_offset", 0.0);
        time_offset = 1e9*lidar_time_offset;        
    }

    void ousterLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr ouster_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<ouster_ros::Point>::Ptr pc_last_ouster(new pcl::PointCloud<ouster_ros::Point>());
        pcl::fromROSMsg(*ouster_msg_in, *pc_last_ouster);
        size_t plsize = pc_last_ouster->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        for (uint i = 1; i < plsize; i++) {
            pt.x = pc_last_ouster->points[i].x;
            pt.y = pc_last_ouster->points[i].y;
            pt.z = pc_last_ouster->points[i].z;
            pt.intensity = float (pc_last_ouster->points[i].t) / float (1e6); // unit: ms
            pt.curvature = 0.1 * pc_last_ouster->points[i].intensity;

            if (pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(ouster_msg_in->header.stamp).nanoseconds() - time_offset;
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        ds_filter_each_scan.setInputCloud(pc_last);
        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(ouster_msg_in->header.stamp).nanoseconds() - time_offset;
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_ouster;
    int64_t time_offset = 0;
};

class Mid70AviaBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  Mid70AviaBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_livox = nh->create_subscription<livox_ros_driver::msg::CustomMsg>(
            this->lidar.topic, 100, std::bind(&Mid70AviaBuff::livoxLidarCallback, this, std::placeholders::_1));
    }

    void livoxLidarCallback(const livox_ros_driver::msg::CustomMsg::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        for (int i = 1; i < plsize; i++) {
            if ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00) {
                pt.x = livox_msg_in->points[i].x;
                pt.y = livox_msg_in->points[i].y;
                pt.z = livox_msg_in->points[i].z;
                pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6); // unit: ms         
                pt.curvature = livox_msg_in->points[i].reflectivity;
                pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        ds_filter_each_scan.setInputCloud(pc_last);
        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<livox_ros_driver::msg::CustomMsg>::SharedPtr pc_subscription_livox;
};

class HAP360Buff : public MappingBase<pcl::PointXYZINormal>
{
public:
    HAP360Buff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_livox = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(
            this->lidar.topic, 100, std::bind(&HAP360Buff::livoxLidarCallback, this, std::placeholders::_1));
    }

    void livoxLidarCallback(livox_ros_driver2::msg::CustomMsg::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        for (int i = 1; i < plsize; i++) {
            if ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00) {
                pt.x = livox_msg_in->points[i].x;
                pt.y = livox_msg_in->points[i].y;
                pt.z = livox_msg_in->points[i].z;
                pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6); // unit: ms         
                pt.curvature = livox_msg_in->points[i].reflectivity;
                pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        ds_filter_each_scan.setInputCloud(pc_last);
        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr pc_subscription_livox;
};

class AviaRespleBuff : public MappingBase<pcl::PointXYZINormal>
{
public:
    AviaRespleBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_livox = nh->create_subscription<livox_interfaces::msg::CustomMsg>(
            this->lidar.topic, 100, std::bind(&AviaRespleBuff::livoxLidarCallback, this, std::placeholders::_1));
    }

    void livoxLidarCallback(livox_interfaces::msg::CustomMsg::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        for (int i = 1; i < plsize; i++) {
            if ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00) {
                pt.x = livox_msg_in->points[i].x;
                pt.y = livox_msg_in->points[i].y;
                pt.z = livox_msg_in->points[i].z;
                pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6); // unit: ms         
                pt.curvature = livox_msg_in->points[i].reflectivity;
                pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        ds_filter_each_scan.setInputCloud(pc_last);
        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<livox_interfaces::msg::CustomMsg>::SharedPtr pc_subscription_livox;
};

class HesaiBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  HesaiBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_hesai = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&HesaiBuff::hesaiLidarCallback, this, std::placeholders::_1));
    }

    void hesaiLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr hesai_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<hesai_ros::Point>::Ptr pc_last_hesai(new pcl::PointCloud<hesai_ros::Point>());
        pcl::fromROSMsg(*hesai_msg_in, *pc_last_hesai);
        size_t plsize = pc_last_hesai->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(hesai_msg_in->header.stamp);
        pcl::PointXYZINormal pt;
        for (uint i = 0; i < plsize; i++) {
            pt.x = pc_last_hesai->points[i].x;
            pt.y = pc_last_hesai->points[i].y;
            pt.z = pc_last_hesai->points[i].z;
            double timestamp_s;
            double timestamp_ns = std::modf(pc_last_hesai->points[i].timestamp, &timestamp_s);
            rclcpp::Time timestamp_ros(static_cast<int32_t>(timestamp_s), static_cast<int32_t>(timestamp_ns * 1.0e9),
                rcl_clock_type_t::RCL_ROS_TIME);
            pt.intensity = (timestamp_ros - timestamp_begin).seconds() * 1.0e3; 
            pt.curvature = pc_last_hesai->points[i].intensity;

            if (pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(hesai_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        ds_filter_each_scan.setInputCloud(pc_last);
        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(hesai_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_hesai;
};

class Mid360BoxiBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  Mid360BoxiBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_mid360 = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&Mid360BoxiBuff::mid360BoxiCallback, this, std::placeholders::_1));
    }

    void mid360BoxiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr livox_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<livox_mid360_boxi::Point>::Ptr pc_last_livox(new pcl::PointCloud<livox_mid360_boxi::Point>());
        pcl::fromROSMsg(*livox_msg_in, *pc_last_livox);
        size_t plsize = pc_last_livox->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(livox_msg_in->header.stamp);
        pcl::PointXYZINormal pt;
        for (uint i = 0; i < plsize; i++) {
            pt.x = pc_last_livox->points[i].x;
            pt.y = pc_last_livox->points[i].y;
            pt.z = pc_last_livox->points[i].z;
            rclcpp::Time timestamp_ros(static_cast<int64_t>(pc_last_livox->points[i].timestamp),
                rcl_clock_type_t::RCL_ROS_TIME);
            pt.intensity = (timestamp_ros - timestamp_begin).seconds() * 1.0e3; 
            pt.curvature = pc_last_livox->points[i].intensity;

            if (pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }
        }
        this->pc_last->header.frame_id = this->frame_id;
        this->pc_last->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->pc_last, *this->pc_last, indices);
        if (this->pc_last->points.empty()) return;
        ds_filter_each_scan.setInputCloud(pc_last);
        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_mid360;
};

class SnailContinentalBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  SnailContinentalBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_snail_conti = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&SnailContinentalBuff::snailContiCallback, this, std::placeholders::_1));
    }

    void snailContiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr conti_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<snail_conti::Point>::Ptr pc_last_conti(new pcl::PointCloud<snail_conti::Point>());
        pcl::fromROSMsg(*conti_msg_in, *pc_last_conti);
        size_t plsize = pc_last_conti->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(conti_msg_in->header.stamp);
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        float max_range = lidar.max_range;
        float min_z = lidar.min_z;
        for (uint i = 0; i < plsize; i++) {
            pt.x = pc_last_conti->points[i].x;
            pt.y = pc_last_conti->points[i].y;
            pt.z = pc_last_conti->points[i].z;
            // double timestamp_s;
            // double timestamp_ns = std::modf(pc_last_conti->points[i].timestamp, &timestamp_s);
            // rclcpp::Time timestamp_ros(static_cast<int32_t>(timestamp_s), static_cast<int32_t>(timestamp_ns * 1.0e9),
            //     rcl_clock_type_t::RCL_ROS_TIME);
            pt.intensity = 0;
            pt.curvature = pc_last_conti->points[i].intensity;
            pt.normal_z = pc_last_conti->points[i].doppler;
            if ((pt.intensity >= 0) && (pt.z > min_z) && (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind)) && (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z < (max_range * max_range)) ) {
            // if(pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }
        }

        int filter_size = pc_last->points.size();
        Eigen::MatrixXd radar_data(filter_size,4);
        const auto& ransac_config = this->radar_ransac_config;

        for(int i=0; i<filter_size; i++){
            const auto & radar_pt = pc_last->points[i];
            const double denom = computeRadarNormalizationDenominator(radar_pt, ransac_config);
            radar_data(i,0) = radar_pt.x / denom;
            radar_data(i,1) = radar_pt.y / denom;
            radar_data(i,2) = radar_pt.z / denom;
            radar_data(i,3) = radar_pt.normal_z;
            // RCLCPP_INFO(rclcpp::get_logger("radar"), "radar data: %f, %f, %f, %f", radar_data(i,0), radar_data(i,1), radar_data(i,2), radar_data(i,3));
        }

        const int N_ransac_points = ransac_config.sample_size;
        const int ransac_iter = computeRadarRansacIterations(ransac_config);
        // int ransac_iter = 100;
        const double inlier_threshold = ransac_config.inlier_threshold;

        Eigen::Vector3d ego_vel;
        Eigen::Vector3d sigma_v_r;

        std::vector<uint> idx(radar_data.rows());
        for (int k = 0; k < radar_data.rows(); ++k)
            idx[k] = k;

        std::random_device rd;
        std::mt19937 g(rd());
        Eigen::MatrixXd H_all(radar_data.rows(), 3);
        H_all.col(0)       = radar_data.col(0);
        H_all.col(1)       = radar_data.col(1);
        H_all.col(2)       = radar_data.col(2);
        const Eigen::VectorXd y_all = radar_data.col(3);

        if (radar_data.rows() >= N_ransac_points)
        {   
            std::vector<uint> inlier_idx_best;
            for (int k = 0; k < ransac_iter; ++k)
            {
                std::shuffle(idx.begin(), idx.end(), g);
                Eigen::MatrixXd radar_data_iter(N_ransac_points, 4);
                Eigen::Vector3d v_r;

                for (int i = 0; i < N_ransac_points; ++i)
                    radar_data_iter.row(i) = radar_data.row(idx.at(i));

                Eigen::MatrixXd H(radar_data_iter.rows(), 3);
                H.col(0)         = radar_data_iter.col(0);
                H.col(1)         = radar_data_iter.col(1);
                H.col(2)         = radar_data_iter.col(2);
                // const Eigen::MatrixXd HTH = H.transpose() * H;

                const Eigen::VectorXd y = radar_data_iter.col(3);

                v_r = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
                const Eigen::VectorXd err = (y_all - H_all * v_r).array().abs();
                std::vector<uint> inlier_idx;
                for (int j = 0; j < err.rows(); ++j)
                    if (err(j) < inlier_threshold)
                        inlier_idx.emplace_back(j);
                if (inlier_idx.size() > inlier_idx_best.size())
                    inlier_idx_best = inlier_idx;                    
            }

            if (!inlier_idx_best.empty())
            {
                Eigen::MatrixXd radar_data_inlier(inlier_idx_best.size(), 4);
                for (uint i = 0; i < inlier_idx_best.size(); ++i)
                    radar_data_inlier.row(i) = radar_data.row(inlier_idx_best.at(i));

                Eigen::MatrixXd H(radar_data_inlier.rows(), 3);
                H.col(0)         = radar_data_inlier.col(0);
                H.col(1)         = radar_data_inlier.col(1);
                H.col(2)         = radar_data_inlier.col(2);
                const Eigen::MatrixXd HTH = H.transpose() * H;

                const Eigen::VectorXd y = radar_data_inlier.col(3);

                // Eigen::JacobiSVD<Eigen::MatrixXd> svd(HTH);
                // auto cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

                // if (std::fabs(cond) < 1.0e3)
                {
                    // if (use_cholesky_instead_of_bdcsvd)
                    //     v_r = (HTH).ldlt().solve(H.transpose() * y);
                    // else
                    ego_vel = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

                    const Eigen::VectorXd e    = H * ego_vel - y;
                    Eigen::MatrixXd P_v_r      = (e.transpose() * e).x() * (HTH).inverse() / (H.rows() - 3);
                    sigma_v_r = Eigen::Vector3d(P_v_r(0, 0), P_v_r(1, 1), P_v_r(2, 2));

                    // const Eigen::Vector3d offset =
                    //     Eigen::Vector3d(sigma_offset_radar_x, sigma_offset_radar_y, sigma_offset_radar_z)
                    //         .array()
                    //         .square();
                    // P_v_r += offset.asDiagonal();

                    // check diagonal for valid estimation result
                    if (sigma_v_r.x() >= 0.0 && sigma_v_r.y() >= 0.0 && sigma_v_r.z() >= 0.0)
                    {
                        sigma_v_r = sigma_v_r.array().sqrt();
                    }
                    else
                    {
                        // sigma_v_r = Eigen::Vector3d::Identity();
                        // std::cerr << "Error: Too high cov" << std::endl;
                        // return;
                    }   
                }

                static_scan->points.clear();
                for (uint i = 0; i < inlier_idx_best.size(); ++i)
                {
                    const auto & radar_pt = pc_last->points[inlier_idx_best.at(i)];
                    static_scan->points.push_back(radar_pt);
                }
            }
        }
        else{
            // std::cerr << "Error: Not enough radar data points for RANSAC" << std::endl;
            return;
        }

        this->static_scan->header.frame_id = this->frame_id;
        this->static_scan->header.stamp = rclcpp::Time(conti_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->static_scan, *this->static_scan, indices);
        if (this->static_scan->points.size() < this->radar_ransac_config.min_static_points) return;
        ds_filter_each_scan.setInputCloud(static_scan);
        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(conti_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_snail_conti;
};

class HerculesContinentalBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  HerculesContinentalBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_hercules_conti = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&HerculesContinentalBuff::herculesContiCallback, this, std::placeholders::_1));
    }

    void herculesContiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr conti_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<hercules_conti::Point>::Ptr pc_last_conti(new pcl::PointCloud<hercules_conti::Point>());
        pcl::fromROSMsg(*conti_msg_in, *pc_last_conti);
        size_t plsize = pc_last_conti->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(conti_msg_in->header.stamp);
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        float max_range = lidar.max_range;
        float min_z = lidar.min_z;
        for (uint i = 0; i < plsize; i++) {
            pt.x = pc_last_conti->points[i].x;
            pt.y = pc_last_conti->points[i].y;
            pt.z = pc_last_conti->points[i].z;
            // double timestamp_s;
            // double timestamp_ns = std::modf(pc_last_conti->points[i].timestamp, &timestamp_s);
            // rclcpp::Time timestamp_ros(static_cast<int32_t>(timestamp_s), static_cast<int32_t>(timestamp_ns * 1.0e9),
            //     rcl_clock_type_t::RCL_ROS_TIME);
            pt.intensity = 0;
            pt.curvature = pc_last_conti->points[i].RCS;
            pt.normal_z = pc_last_conti->points[i].v;
            if ((pt.intensity >= 0) && (pt.z > min_z) && (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind)) && (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z < (max_range * max_range)) ) {
            // if(pt.intensity >= 0) {
                this->pc_last->points.push_back(pt);
            }
        }

             
        int filter_size = pc_last->points.size();
        Eigen::MatrixXd radar_data(filter_size,4);
        const auto& ransac_config = this->radar_ransac_config;

        for(int i=0; i<filter_size; i++){
            const auto & radar_pt = pc_last->points[i];
            const double denom = computeRadarNormalizationDenominator(radar_pt, ransac_config);
            radar_data(i,0) = radar_pt.x / denom;
            radar_data(i,1) = radar_pt.y / denom;
            radar_data(i,2) = radar_pt.z / denom;
            radar_data(i,3) = radar_pt.normal_z;
            // RCLCPP_INFO(rclcpp::get_logger("radar"), "radar data: %f, %f, %f, %f", radar_data(i,0), radar_data(i,1), radar_data(i,2), radar_data(i,3));
        }

        const int N_ransac_points = ransac_config.sample_size;
        const int ransac_iter = computeRadarRansacIterations(ransac_config);
        // int ransac_iter = 100;
        const double inlier_threshold = ransac_config.inlier_threshold;

        Eigen::Vector3d ego_vel;
        Eigen::Vector3d sigma_v_r;

        std::vector<uint> idx(radar_data.rows());
        for (int k = 0; k < radar_data.rows(); ++k)
            idx[k] = k;

        std::random_device rd;
        std::mt19937 g(rd());
        Eigen::MatrixXd H_all(radar_data.rows(), 3);
        H_all.col(0)       = radar_data.col(0);
        H_all.col(1)       = radar_data.col(1);
        H_all.col(2)       = radar_data.col(2);
        const Eigen::VectorXd y_all = radar_data.col(3);

        if (radar_data.rows() >= N_ransac_points)
        {   
            std::vector<uint> inlier_idx_best;
            for (int k = 0; k < ransac_iter; ++k)
            {
                std::shuffle(idx.begin(), idx.end(), g);
                Eigen::MatrixXd radar_data_iter(N_ransac_points, 4);
                Eigen::Vector3d v_r;

                for (int i = 0; i < N_ransac_points; ++i)
                    radar_data_iter.row(i) = radar_data.row(idx.at(i));

                Eigen::MatrixXd H(radar_data_iter.rows(), 3);
                H.col(0)         = radar_data_iter.col(0);
                H.col(1)         = radar_data_iter.col(1);
                H.col(2)         = radar_data_iter.col(2);
                // const Eigen::MatrixXd HTH = H.transpose() * H;

                const Eigen::VectorXd y = radar_data_iter.col(3);

                v_r = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
                const Eigen::VectorXd err = (y_all - H_all * v_r).array().abs();
                std::vector<uint> inlier_idx;
                for (int j = 0; j < err.rows(); ++j)
                    if (err(j) < inlier_threshold)
                        inlier_idx.emplace_back(j);
                if (inlier_idx.size() > inlier_idx_best.size())
                    inlier_idx_best = inlier_idx;                    
            }

            if (!inlier_idx_best.empty())
            {
                Eigen::MatrixXd radar_data_inlier(inlier_idx_best.size(), 4);
                for (uint i = 0; i < inlier_idx_best.size(); ++i)
                    radar_data_inlier.row(i) = radar_data.row(inlier_idx_best.at(i));

                Eigen::MatrixXd H(radar_data_inlier.rows(), 3);
                H.col(0)         = radar_data_inlier.col(0);
                H.col(1)         = radar_data_inlier.col(1);
                H.col(2)         = radar_data_inlier.col(2);
                const Eigen::MatrixXd HTH = H.transpose() * H;

                const Eigen::VectorXd y = radar_data_inlier.col(3);

                // Eigen::JacobiSVD<Eigen::MatrixXd> svd(HTH);
                // auto cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

                // if (std::fabs(cond) < 1.0e3)
                {
                    // if (use_cholesky_instead_of_bdcsvd)
                    //     v_r = (HTH).ldlt().solve(H.transpose() * y);
                    // else
                    ego_vel = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

                    const Eigen::VectorXd e    = H * ego_vel - y;
                    Eigen::MatrixXd P_v_r      = (e.transpose() * e).x() * (HTH).inverse() / (H.rows() - 3);
                    sigma_v_r = Eigen::Vector3d(P_v_r(0, 0), P_v_r(1, 1), P_v_r(2, 2));

                    // const Eigen::Vector3d offset =
                    //     Eigen::Vector3d(sigma_offset_radar_x, sigma_offset_radar_y, sigma_offset_radar_z)
                    //         .array()
                    //         .square();
                    // P_v_r += offset.asDiagonal();

                    // check diagonal for valid estimation result
                    if (sigma_v_r.x() >= 0.0 && sigma_v_r.y() >= 0.0 && sigma_v_r.z() >= 0.0)
                    {
                        sigma_v_r = sigma_v_r.array().sqrt();
                    }
                    else
                    {
                        // sigma_v_r = Eigen::Vector3d::Identity();
                        // std::cerr << "Error: Too high cov" << std::endl;
                        // return;
                    }   
                }

                static_scan->points.clear();
                for (uint i = 0; i < inlier_idx_best.size(); ++i)
                {
                    const auto & radar_pt = pc_last->points[inlier_idx_best.at(i)];
                    static_scan->points.push_back(radar_pt);
                }
            }
        }
        else{
            // std::cerr << "Error: Not enough radar data points for RANSAC" << std::endl;
            return;
        }

        this->static_scan->header.frame_id = this->frame_id;
        this->static_scan->header.stamp = rclcpp::Time(conti_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->static_scan, *this->static_scan, indices);
        if (this->static_scan->points.size() < this->radar_ransac_config.min_static_points) return;
        ds_filter_each_scan.setInputCloud(static_scan);

        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(conti_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_hercules_conti;
};

class HuginBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  HuginBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_hugin = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&HuginBuff::huginCallback, this, std::placeholders::_1));
    }

    void huginCallback(const sensor_msgs::msg::PointCloud2::SharedPtr hugin_msg_in)
    {
        this->pc_last->clear();
        pcl::PointCloud<hugin::Point>::Ptr pc_last_hugin(new pcl::PointCloud<hugin::Point>());
        pcl::fromROSMsg(*hugin_msg_in, *pc_last_hugin);
        const size_t plsize = pc_last_hugin->size();
        if (plsize == 0) return;
        this->pc_last->reserve(plsize);
        pcl::PointXYZINormal pt;
        const float blind = lidar.blind;
        const float max_range = lidar.max_range;
        const float min_z = lidar.min_z;
        for (uint i = 0; i < plsize; i++) {
            pt.x = pc_last_hugin->points[i].x;
            pt.y = pc_last_hugin->points[i].y;
            pt.z = pc_last_hugin->points[i].z;
            pt.intensity = 0;
            pt.curvature = pc_last_hugin->points[i].power;
            pt.normal_z = pc_last_hugin->points[i].doppler;
            if ((pt.intensity >= 0) && (pt.z > min_z) &&
                (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z > (blind * blind)) &&
                (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z < (max_range * max_range))) {
                this->pc_last->points.push_back(pt);
            }
        }

        const int filter_size = pc_last->points.size();
        Eigen::MatrixXd radar_data(filter_size,4);
        const auto& ransac_config = this->radar_ransac_config;

        for (int i = 0; i < filter_size; i++) {
            const auto & radar_pt = pc_last->points[i];
            const double denom = computeRadarNormalizationDenominator(radar_pt, ransac_config);
            radar_data(i,0) = radar_pt.x / denom;
            radar_data(i,1) = radar_pt.y / denom;
            radar_data(i,2) = radar_pt.z / denom;
            radar_data(i,3) = radar_pt.normal_z;
        }

        const int N_ransac_points = ransac_config.sample_size;
        const int ransac_iter = computeRadarRansacIterations(ransac_config);
        const double inlier_threshold = ransac_config.inlier_threshold;

        Eigen::Vector3d ego_vel;
        Eigen::Vector3d sigma_v_r;

        std::vector<uint> idx(radar_data.rows());
        for (int k = 0; k < radar_data.rows(); ++k)
            idx[k] = k;

        std::random_device rd;
        std::mt19937 g(rd());
        Eigen::MatrixXd H_all(radar_data.rows(), 3);
        H_all.col(0) = radar_data.col(0);
        H_all.col(1) = radar_data.col(1);
        H_all.col(2) = radar_data.col(2);
        const Eigen::VectorXd y_all = radar_data.col(3);

        if (radar_data.rows() >= N_ransac_points)
        {
            std::vector<uint> inlier_idx_best;
            for (int k = 0; k < ransac_iter; ++k)
            {
                std::shuffle(idx.begin(), idx.end(), g);
                Eigen::MatrixXd radar_data_iter(N_ransac_points, 4);
                Eigen::Vector3d v_r;

                for (int i = 0; i < N_ransac_points; ++i)
                    radar_data_iter.row(i) = radar_data.row(idx.at(i));

                Eigen::MatrixXd H(radar_data_iter.rows(), 3);
                H.col(0) = radar_data_iter.col(0);
                H.col(1) = radar_data_iter.col(1);
                H.col(2) = radar_data_iter.col(2);

                const Eigen::VectorXd y = radar_data_iter.col(3);

                v_r = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
                const Eigen::VectorXd err = (y_all - H_all * v_r).array().abs();
                std::vector<uint> inlier_idx;
                for (int j = 0; j < err.rows(); ++j)
                    if (err(j) < inlier_threshold)
                        inlier_idx.emplace_back(j);
                if (inlier_idx.size() > inlier_idx_best.size())
                    inlier_idx_best = inlier_idx;
            }

            if (!inlier_idx_best.empty())
            {
                Eigen::MatrixXd radar_data_inlier(inlier_idx_best.size(), 4);
                for (uint i = 0; i < inlier_idx_best.size(); ++i)
                    radar_data_inlier.row(i) = radar_data.row(inlier_idx_best.at(i));

                Eigen::MatrixXd H(radar_data_inlier.rows(), 3);
                H.col(0) = radar_data_inlier.col(0);
                H.col(1) = radar_data_inlier.col(1);
                H.col(2) = radar_data_inlier.col(2);
                const Eigen::MatrixXd HTH = H.transpose() * H;

                const Eigen::VectorXd y = radar_data_inlier.col(3);

                ego_vel = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

                const Eigen::VectorXd e = H * ego_vel - y;
                Eigen::MatrixXd P_v_r = (e.transpose() * e).x() * (HTH).inverse() / (H.rows() - 3);
                sigma_v_r = Eigen::Vector3d(P_v_r(0, 0), P_v_r(1, 1), P_v_r(2, 2));

                if (sigma_v_r.x() >= 0.0 && sigma_v_r.y() >= 0.0 && sigma_v_r.z() >= 0.0)
                {
                    sigma_v_r = sigma_v_r.array().sqrt();
                }

                static_scan->points.clear();
                for (uint i = 0; i < inlier_idx_best.size(); ++i)
                {
                    const auto & radar_pt = pc_last->points[inlier_idx_best.at(i)];
                    static_scan->points.push_back(radar_pt);
                }
            }
        }
        else{
            return;
        }

        this->static_scan->header.frame_id = this->frame_id;
        this->static_scan->header.stamp = rclcpp::Time(hugin_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->static_scan, *this->static_scan, indices);
        if (this->static_scan->points.size() < this->radar_ransac_config.min_static_points) return;
        ds_filter_each_scan.setInputCloud(static_scan);
        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(hugin_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_hugin;
};

class HkustContinentalBuff : public MappingBase<pcl::PointXYZINormal>
{
  public:
  HkustContinentalBuff(rclcpp::Node::SharedPtr &nh, const LidarConfig& lidar_config, const RadarRansacConfig& ransac_config) : MappingBase<pcl::PointXYZINormal>(nh, lidar_config, ransac_config)
    {
        pc_subscription_hkust_conti = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
            this->lidar.topic, 100, std::bind(&HkustContinentalBuff::hkustContiCallback, this, std::placeholders::_1));
    }

    void hkustContiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr conti_msg_in)
    {
        this->pc_last->clear();
        std::string name = "HkustContinental";
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());     
        // pcl::PointCloud<hkust_conti::Point>::Ptr pc_last_conti(new pcl::PointCloud<hkust_conti::Point>());
        // pcl::fromROSMsg(*conti_msg_in, *pc_last_conti);
        // size_t plsize = pc_last_conti->size();
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "HKUST Continental point cloud size: %zu", plsize);
        // if (plsize == 0) return;
        const int pointBytes = static_cast<int>(conti_msg_in->point_step);

        int offsetAzimuth = 0, offsetAzimuthSTD = 0;
        int offsetElevation = 0, offsetElevationSTD = 0;
        int offsetRange = 0, offsetRangeSTD = 0;
        int offsetVelocity = 0, offsetVelocitySTD = 0;
        int offsetRCS = 0;

        // 필드 오프셋 찾기
        for (const auto &f : conti_msg_in->fields) {
            if (f.name == "azimuth")        offsetAzimuth      = f.offset;
            else if (f.name == "azimuthSTD") offsetAzimuthSTD   = f.offset;
            else if (f.name == "elevation")  offsetElevation    = f.offset;
            else if (f.name == "elevationSTD") offsetElevationSTD = f.offset;
            else if (f.name == "range")      offsetRange        = f.offset;
            else if (f.name == "rangeSTD")   offsetRangeSTD     = f.offset;
            else if (f.name == "velocity")   offsetVelocity     = f.offset;
            else if (f.name == "velocitySTD") offsetVelocitySTD  = f.offset;
            else if (f.name == "rcs")        offsetRCS          = f.offset;
        }

        size_t plsize = conti_msg_in->data.size() / pointBytes;
        RCLCPP_INFO(rclcpp::get_logger("radar"), "HKUST Continental point cloud size: %zu", plsize);
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(conti_msg_in->header.stamp);
        int64_t time_begin = timestamp_begin.nanoseconds();
        static int64_t last_t_ns = time_begin;   
        int64_t max_ofs_ns = 0;         
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        float max_range = lidar.max_range;
        float min_z = lidar.min_z;
        const uint8_t *data_ptr = conti_msg_in->data.data();
        for (unsigned int i = 0; i < plsize; ++i) {
            const uint8_t *pt_base = data_ptr + i * pointBytes;
            double azimuth     = *reinterpret_cast<const float*>(pt_base + offsetAzimuth);
            double elevation   = *reinterpret_cast<const float*>(pt_base + offsetElevation);
            double range       = *reinterpret_cast<const float*>(pt_base + offsetRange);
            pt.x = range * cos(azimuth) * cos(elevation);
            pt.y = range * sin(azimuth) * cos(elevation);
            pt.z = range * sin(elevation);
            pt.intensity = 0;
            // TODO : raw intensity is in [0, 255] unit: dB
            // TODO : need type casting?
            pt.curvature = *reinterpret_cast<const float*>(pt_base + offsetRCS) + 128.0; // make positive
            pt.normal_z = *reinterpret_cast<const float*>(pt_base + offsetVelocity);
                

            if ((pt.intensity >= 0) && (pt.z > min_z) && (range * range > (blind * blind)) && (range * range < (max_range * max_range)) ) {
                // int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                // max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                pc_last->points.push_back(pt);
            }
            
        }
 
        int filter_size = pc_last->points.size();
        Eigen::MatrixXd radar_data(filter_size,4);
        const auto& ransac_config = this->radar_ransac_config;

        for(int i=0; i<filter_size; i++){
            const auto & radar_pt = pc_last->points[i];
            const double denom = computeRadarNormalizationDenominator(radar_pt, ransac_config);
            radar_data(i,0) = radar_pt.x / denom;
            radar_data(i,1) = radar_pt.y / denom;
            radar_data(i,2) = radar_pt.z / denom;
            radar_data(i,3) = radar_pt.normal_z;
            // RCLCPP_INFO(rclcpp::get_logger("radar"), "radar data: %f, %f, %f, %f", radar_data(i,0), radar_data(i,1), radar_data(i,2), radar_data(i,3));
        }

        const int N_ransac_points = ransac_config.sample_size;
        const int ransac_iter = computeRadarRansacIterations(ransac_config);
        // int ransac_iter = 100;
        const double inlier_threshold = ransac_config.inlier_threshold;

        Eigen::Vector3d ego_vel;
        Eigen::Vector3d sigma_v_r;

        std::vector<uint> idx(radar_data.rows());
        for (int k = 0; k < radar_data.rows(); ++k)
            idx[k] = k;

        std::random_device rd;
        std::mt19937 g(rd());
        Eigen::MatrixXd H_all(radar_data.rows(), 3);
        H_all.col(0)       = radar_data.col(0);
        H_all.col(1)       = radar_data.col(1);
        H_all.col(2)       = radar_data.col(2);
        const Eigen::VectorXd y_all = radar_data.col(3);

        if (radar_data.rows() >= N_ransac_points)
        {   
            std::vector<uint> inlier_idx_best;
            for (int k = 0; k < ransac_iter; ++k)
            {
                std::shuffle(idx.begin(), idx.end(), g);
                Eigen::MatrixXd radar_data_iter(N_ransac_points, 4);
                Eigen::Vector3d v_r;

                for (int i = 0; i < N_ransac_points; ++i)
                    radar_data_iter.row(i) = radar_data.row(idx.at(i));

                Eigen::MatrixXd H(radar_data_iter.rows(), 3);
                H.col(0)         = radar_data_iter.col(0);
                H.col(1)         = radar_data_iter.col(1);
                H.col(2)         = radar_data_iter.col(2);
                // const Eigen::MatrixXd HTH = H.transpose() * H;

                const Eigen::VectorXd y = radar_data_iter.col(3);

                v_r = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
                const Eigen::VectorXd err = (y_all - H_all * v_r).array().abs();
                std::vector<uint> inlier_idx;
                for (int j = 0; j < err.rows(); ++j)
                    if (err(j) < inlier_threshold)
                        inlier_idx.emplace_back(j);
                if (inlier_idx.size() > inlier_idx_best.size())
                    inlier_idx_best = inlier_idx;                    
            }

            if (!inlier_idx_best.empty())
            {
                Eigen::MatrixXd radar_data_inlier(inlier_idx_best.size(), 4);
                for (uint i = 0; i < inlier_idx_best.size(); ++i)
                    radar_data_inlier.row(i) = radar_data.row(inlier_idx_best.at(i));

                Eigen::MatrixXd H(radar_data_inlier.rows(), 3);
                H.col(0)         = radar_data_inlier.col(0);
                H.col(1)         = radar_data_inlier.col(1);
                H.col(2)         = radar_data_inlier.col(2);
                const Eigen::MatrixXd HTH = H.transpose() * H;

                const Eigen::VectorXd y = radar_data_inlier.col(3);

                // Eigen::JacobiSVD<Eigen::MatrixXd> svd(HTH);
                // auto cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);

                // if (std::fabs(cond) < 1.0e3)
                {
                    // if (use_cholesky_instead_of_bdcsvd)
                    //     v_r = (HTH).ldlt().solve(H.transpose() * y);
                    // else
                    ego_vel = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

                    const Eigen::VectorXd e    = H * ego_vel - y;
                    Eigen::MatrixXd P_v_r      = (e.transpose() * e).x() * (HTH).inverse() / (H.rows() - 3);
                    sigma_v_r = Eigen::Vector3d(P_v_r(0, 0), P_v_r(1, 1), P_v_r(2, 2));

                    // const Eigen::Vector3d offset =
                    //     Eigen::Vector3d(sigma_offset_radar_x, sigma_offset_radar_y, sigma_offset_radar_z)
                    //         .array()
                    //         .square();
                    // P_v_r += offset.asDiagonal();

                    // check diagonal for valid estimation result
                    if (sigma_v_r.x() >= 0.0 && sigma_v_r.y() >= 0.0 && sigma_v_r.z() >= 0.0)
                    {
                        sigma_v_r = sigma_v_r.array().sqrt();
                    }
                    else
                    {
                        // sigma_v_r = Eigen::Vector3d::Identity();
                        // std::cerr << "Error: Too high cov" << std::endl;
                        // return;
                    }   
                }

                static_scan->points.clear();
                for (uint i = 0; i < inlier_idx_best.size(); ++i)
                {
                    const auto & radar_pt = pc_last->points[inlier_idx_best.at(i)];
                    static_scan->points.push_back(radar_pt);
                }
            }
        }
        else{
            // std::cerr << "Error: Not enough radar data points for RANSAC" << std::endl;
            return;
        }

        this->static_scan->header.frame_id = this->frame_id;
        this->static_scan->header.stamp = rclcpp::Time(conti_msg_in->header.stamp).nanoseconds();
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*this->static_scan, *this->static_scan, indices);
        if (this->static_scan->points.size() < this->radar_ransac_config.min_static_points) return;
        ds_filter_each_scan.setInputCloud(static_scan);

        this->pc_last_ds->clear();
        ds_filter_each_scan.filter(*this->pc_last_ds);
        pc_last_ds->header.frame_id = this->frame_id;
        pc_last_ds->header.stamp = rclcpp::Time(conti_msg_in->header.stamp).nanoseconds();
        mtx.lock();
        this->pc_L_buff.push_back(*pc_last_ds);
        mtx.unlock();
    }

  private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_subscription_hkust_conti;
};



class Mapping
{

public:

    Mapping(rclcpp::Node::SharedPtr &nh, std::vector<MappingBase<pcl::PointXYZINormal>*>& mappings)
    {
        visualize_gt = CommonUtils::readParam<bool>(nh, "visualize_gt", true);
        sub_start = nh->create_subscription<std_msgs::msg::Int64>("/start_time", 100, std::bind(&Mapping::startCallBack, this, std::placeholders::_1));
        spl_window_st_ns = 0;
        sub_est = nh->create_subscription<estimate_msgs::msg::Estimate>("/est_window", 10000, std::bind(&Mapping::getEstCallback, this, std::placeholders::_1));
        pub_path = nh->create_publisher<nav_msgs::msg::Path>("traj_path", 100);
        if (visualize_gt) {
            sub_gt_odom = nh->create_subscription<nav_msgs::msg::Odometry>("/gt_odometry", 1000, std::bind(&Mapping::getGtOdomCallback, this, std::placeholders::_1));
            pub_gt_path = nh->create_publisher<nav_msgs::msg::Path>("/gt_path", 100);
        }
        pub_knots = nh->create_publisher<sensor_msgs::msg::PointCloud>("active_control_points",20);
        opt_old_path.header.frame_id = odom_id;
        gt_path.header.frame_id = odom_id;
        vis_maps = mappings;
        pub_odom = nh->create_publisher<nav_msgs::msg::Odometry>("odom_vis", 500);
        br = std::make_shared<tf2_ros::TransformBroadcaster>(nh);
    }

    void lock_mappings() {
        for (auto vis_map : vis_maps) {
            vis_map->mtx.lock();
        }
    }

    void unlock_mappings() {
        for (auto vis_map : vis_maps) {
            vis_map->mtx.unlock();
        }
    }    

    void process() {
        rclcpp::Rate rate(100);
        int64_t num_knot = 0;
        while (true) {
            if (if_init_succeed && spline_global.numKnots() > num_knot) {
                lock_mappings();
                publishPath();
                displayControlPoints();
                pubOdom();
                num_knot = spline_global.numKnots();
                unlock_mappings();
            }
            if (!if_init_succeed) {
                rate.sleep();
                continue;
            }
            for (const auto vis_map : vis_maps) {
                vis_map->processScan(&spline_global, spl_window_st_ns);  
            }                     
        }
    }

private:
    std::string node_name = "Mapping";
    int64_t spl_window_st_ns;
    SplineState spline_global;
    rclcpp::Subscription<estimate_msgs::msg::Estimate>::SharedPtr sub_est;
    rclcpp::Subscription<std_msgs::msg::Int64>::SharedPtr sub_start;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_gt_odom;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    nav_msgs::msg::Path opt_old_path;
    nav_msgs::msg::Path gt_path;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud>::SharedPtr pub_knots;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_gt_path;
    std::vector<MappingBase<pcl::PointXYZINormal>*> vis_maps;
    const std::string frame_id = "base_link";
    const std::string gt_frame_id = "gt_base_link";
    const std::string odom_id = "odom";
    std::shared_ptr<tf2_ros::TransformBroadcaster> br;
    bool if_init_succeed = false;
    bool visualize_gt = true;
    std::mutex m_spline;    

    void displayControlPoints()
    {
        sensor_msgs::msg::PointCloud points_msg;
        points_msg.header.frame_id = odom_id;
        points_msg.header.stamp = rclcpp::Time(spline_global.minTimeNs());
        for (int64_t i = spline_global.numKnots() - 4; i < spline_global.numKnots(); i++) {
            points_msg.points.push_back(CommonUtils::getPointMsg(spline_global.getKnotPos(i)));
        }
        pub_knots->publish(points_msg);
    }

    void getEstCallback(const estimate_msgs::msg::Estimate::SharedPtr est_msg)
    {
        if (!if_init_succeed) {
            return;
        }
        estimate_msgs::msg::Spline spline_msg = est_msg->spline;
        SplineState spline_w;
        
        spline_w.init(spline_msg.dt, 0, spline_msg.start_t, spline_msg.start_idx);
        for(const auto& knot : spline_msg.knots) {
            Eigen::Vector3d pos(knot.position.x, knot.position.y, knot.position.z);
            Eigen::Vector3d quat_del(knot.orientation_del.x, knot.orientation_del.y, knot.orientation_del.z);
            spline_w.addOneStateKnot(pos, quat_del);
        }
        Eigen::Quaterniond q_idle0 = Eigen::Quaterniond(spline_msg.start_q.w, spline_msg.start_q.x, spline_msg.start_q.y, spline_msg.start_q.z);
        for (int i = 0; i < 3; i++) {
            estimate_msgs::msg::Knot idle = spline_msg.idles[i];
            Eigen::Vector3d t_idle(idle.position.x, idle.position.y, idle.position.z);
            Eigen::Vector3d quat_idle(idle.orientation_del.x, idle.orientation_del.y, idle.orientation_del.z);
            spline_w.setIdles(i, t_idle, quat_idle, q_idle0);
        }
        lock_mappings();
        spl_window_st_ns = spline_msg.start_t - spline_msg.dt; 
        spline_global.setTimeIntervalNs(spline_msg.dt);
        spline_global.updateKnots(&spline_w);
        unlock_mappings();
    }

    void pubOdom()
    {
        if (opt_old_path.poses.empty()) {
            return;
        }
        nav_msgs::msg::Odometry odom_msg;
        geometry_msgs::msg::PoseStamped odom_pose = opt_old_path.poses.back();
        odom_msg.header.stamp = rclcpp::Time(odom_pose.header.stamp);
        odom_msg.header.frame_id = odom_id;
        odom_msg.child_frame_id = frame_id;
        odom_msg.pose.pose = odom_pose.pose;
        // pub_odom->publish(odom_msg);      
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = odom_msg.header.stamp;
        transformStamped.header.frame_id = odom_id;
        transformStamped.child_frame_id = frame_id;
        transformStamped.transform.translation.x = odom_pose.pose.position.x;
        transformStamped.transform.translation.y = odom_pose.pose.position.y;
        transformStamped.transform.translation.z = odom_pose.pose.position.z;
        transformStamped.transform.rotation = odom_pose.pose.orientation;
        br->sendTransform(transformStamped);
        transformStamped.header.stamp = odom_msg.header.stamp;
        transformStamped.header.frame_id = frame_id;
        transformStamped.child_frame_id = "imu";
        transformStamped.transform.translation.x = 0;
        transformStamped.transform.translation.y = 0;
        transformStamped.transform.translation.z = 0;
        transformStamped.transform.rotation.w = 1;
        transformStamped.transform.rotation.x = 0;
        transformStamped.transform.rotation.y = 0;
        transformStamped.transform.rotation.z = 0;
        br->sendTransform(transformStamped);

        transformStamped.header.stamp = odom_msg.header.stamp;
        transformStamped.header.frame_id = frame_id;
        transformStamped.child_frame_id = "lidar";
        transformStamped.transform.translation.x = 0;
        transformStamped.transform.translation.y = 0;
        transformStamped.transform.translation.z = 0;
        transformStamped.transform.rotation.w = 1;
        transformStamped.transform.rotation.x = 0;
        transformStamped.transform.rotation.y = 0;
        transformStamped.transform.rotation.z = 0;
        br->sendTransform(transformStamped);
    }

    void getGtOdomCallback(const nav_msgs::msg::Odometry::SharedPtr gt_odom_msg)
    {
        if (!visualize_gt || !pub_gt_path) {
            return;
        }

        geometry_msgs::msg::PoseStamped gt_pose_msg;
        gt_pose_msg.header = gt_odom_msg->header;
        gt_pose_msg.pose = gt_odom_msg->pose.pose;

        if (!gt_path.poses.empty() &&
            rclcpp::Time(gt_path.poses.back().header.stamp).nanoseconds() ==
                rclcpp::Time(gt_pose_msg.header.stamp).nanoseconds()) {
            gt_path.poses.back() = gt_pose_msg;
        } else {
            gt_path.poses.push_back(gt_pose_msg);
        }
        gt_path.header.stamp = gt_pose_msg.header.stamp;
        pub_gt_path->publish(gt_path);

        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header = gt_odom_msg->header;
        transformStamped.header.frame_id = odom_id;
        transformStamped.child_frame_id = gt_frame_id;
        transformStamped.transform.translation.x = gt_pose_msg.pose.position.x;
        transformStamped.transform.translation.y = gt_pose_msg.pose.position.y;
        transformStamped.transform.translation.z = gt_pose_msg.pose.position.z;
        transformStamped.transform.rotation = gt_pose_msg.pose.orientation;
        br->sendTransform(transformStamped);
    }

    void startCallBack(const std_msgs::msg::Int64::SharedPtr start_time_msg)
    {
        int64_t bag_start_time = start_time_msg->data;
        spline_global.init(0, 0, bag_start_time, 0);
        if_init_succeed = true;
    }

    void publishPath() {
        if (!if_init_succeed || spline_global.numKnots() <= 4) {
            return;
        }
        static int64_t t_ns = spline_global.minTimeNs();
        while (t_ns < std::min(spl_window_st_ns, spline_global.maxTimeNs())) {
            Eigen::Quaterniond orient_interp;
            Eigen::Vector3d t_interp = spline_global.itpPosition(t_ns);
            spline_global.itpQuaternion(t_ns, &orient_interp);
            opt_old_path.poses.push_back(CommonUtils::pose2msg(t_ns, t_interp, orient_interp));
            t_ns += 1e8;
        }
        pub_path->publish(opt_old_path);

        // Eigen::Quaterniond orient;
        // Eigen::Vector3d pos = spline_global.itpPosition(t_ns);
        // spline_global.itpQuaternion(t_ns, &orient);

        // std::ofstream result_file_ns("sample.txt", std::ios::app);
        // result_file_ns.setf(std::ios::fixed, std::ios::floatfield);
        // result_file_ns.precision(9);

        // result_file_ns << t_ns << ' '
        //             << pos.x() << ' ' << pos.y() << ' ' << pos.z() << ' '
        //             << orient.x() << ' ' << orient.y() << ' ' << orient.z() << ' ' << orient.w()
        //             << '\n';
        // result_file_ns.close();
    }         

};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto nh = rclcpp::Node::make_shared("Mapping");
    struct ConfiguredLidar {
        LidarConfig lidar;
        std::string prefix;
    };

    std::vector<ConfiguredLidar> lidars;
    auto lidar_names = nh->declare_parameter<std::vector<std::string>>("lidars", std::vector<std::string>());
    assert(nh->get_parameter({"lidars"}, lidar_names));
    if (lidar_names.empty()) {
        lidars.push_back(ConfiguredLidar{LidarConfig(nh, ""), ""});
    } else {
        for (const auto& lidar_name : lidar_names) {
            const std::string lidar_prefix = lidar_name + ".";
            lidars.push_back(ConfiguredLidar{LidarConfig(nh, lidar_prefix), lidar_prefix});
        }
    }
    std::vector<MappingBase<pcl::PointXYZINormal>*> buffs;
    for (const auto& configured_lidar : lidars) {
        const auto& lidar = configured_lidar.lidar;
        const auto radar_ransac_config = loadRadarRansacConfig(nh, configured_lidar.prefix, lidar.type);
        if (!lidar.type.compare("Ouster")) {
            buffs.push_back(new OusterBuff(nh, lidar, radar_ransac_config));
        } else if (!lidar.type.compare("Mid70Avia")) {
            buffs.push_back(new Mid70AviaBuff(nh, lidar, radar_ransac_config));
        } else if (!lidar.type.compare("HAP360")) {
            buffs.push_back(new HAP360Buff(nh, lidar, radar_ransac_config));    
        } else if (!lidar.type.compare("AviaResple")) {
            buffs.push_back(new AviaRespleBuff(nh, lidar, radar_ransac_config));
        } else if (!lidar.type.compare("Hesai")) {
            buffs.push_back(new HesaiBuff(nh, lidar, radar_ransac_config));
        } else if (!lidar.type.compare("Mid360Boxi")) {
            buffs.push_back(new Mid360BoxiBuff(nh, lidar, radar_ransac_config));
        } else if (!lidar.type.compare("SnailContinental")) {
            buffs.push_back(new SnailContinentalBuff(nh, lidar, radar_ransac_config));
        } else if (!lidar.type.compare("Hugin")) {
            buffs.push_back(new HuginBuff(nh, lidar, radar_ransac_config));
        } else if (!lidar.type.compare("HerculesContinental")) {
            buffs.push_back(new HerculesContinentalBuff(nh, lidar, radar_ransac_config));
        } else if (!lidar.type.compare("HkustContinental")) {
            buffs.push_back(new HkustContinentalBuff(nh, lidar, radar_ransac_config));
        } else {
            exit(1);
        }
    }
    Mapping mapping(nh, buffs);
    std::thread mappingThread{&Mapping::process, &mapping};
    rclcpp::spin(nh);
    mappingThread.join();
    rclcpp::shutdown();
}
