#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <std_msgs/msg/int64.hpp>
#include <std_msgs/msg/bool.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <thread>
#include <mutex>
#include <boost/make_shared.hpp>
#include <rclcpp/service.hpp>
#include <std_srvs/srv/empty.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "livox_interfaces/msg/custom_msg.hpp"
#include "livox_ros_driver/msg/custom_msg.hpp"
#include "livox_ros_driver2/msg/custom_msg.hpp"
#include "estimate_msgs/msg/calib.hpp"
#include "estimate_msgs/msg/spline.hpp"
#include "estimate_msgs/msg/estimate.hpp"
#include "Estimator.h"

#include "std_msgs/msg/string.hpp"

#include <random>
#include <cmath>

KD_TREE<pcl::PointXYZINormal> ikdtree;
KD_TREE<pcl::PointXYZINormal> ikdtree_gt;
VoxelMapManagerPtr voxelmap_manager;
std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;

pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_map_tot;
pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_map_gtpose;



using TumEntry = std::pair<int64_t, std::pair<Eigen::Vector3d, Eigen::Quaterniond>>;
// TumEntry: first=timestamp(ns, i64), second.first=translation, second.second=quaternion(wxyz)
std::vector<TumEntry> gt_pose;

static int64_t unit_scale_ns(const std::string& u) {
    if (u == "s")  return 1000000000LL;
    if (u == "ms") return 1000000LL;
    if (u == "us") return 1000LL;
    if (u == "ns") return 1LL;
    throw std::runtime_error("unit must be one of: s|ms|us|ns");
}

std::vector<TumEntry> load_tum_as_ns(const std::string& path, const std::string& unit) {
    const int64_t scale = unit_scale_ns(unit);
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("cannot open: " + path);

    std::vector<TumEntry> out;
    std::string line;

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        std::string ts_tok; double tx,ty,tz,qx,qy,qz,qw;
        if (!(iss >> ts_tok >> tx >> ty >> tz >> qx >> qy >> qz >> qw)) continue;

        // time -> ns (i64)
        int64_t t_ns = 0;
        if (ts_tok.find_first_of(".eE") != std::string::npos) {
            // fractional -> seconds -> ns
            long double secs = std::stold(ts_tok);
            t_ns = static_cast<int64_t>(std::llround(secs * 1000000000.0L));
        } else {
            // integer in given unit
            int64_t v = std::stoll(ts_tok);
            t_ns = v * scale;
        }
        // RCLCPP_INFO(rclcpp::get_logger("load_tum_as_ns"), "Loaded GT pose with timestamp: %ld", t_ns);

        Eigen::Quaterniond q(qw, qx, qy, qz);
        q.normalize();
        out.emplace_back(t_ns, std::make_pair(Eigen::Vector3d(tx,ty,tz), q));
    }
    return out;
}

static bool interpolate_gt_pose_ns(const std::vector<TumEntry>& gt_poses,
                                   int64_t t_ns,
                                   Eigen::Vector3d& pos_gt,
                                   Eigen::Quaterniond& quat_gt) {
    if (gt_poses.empty()) {
        return false;
    }

    auto it = std::lower_bound(gt_poses.begin(), gt_poses.end(), t_ns, [](const TumEntry& entry, int64_t t) {
        return entry.first < t;
    });

    if (it == gt_poses.end()) {
        pos_gt = gt_poses.back().second.first;
        quat_gt = gt_poses.back().second.second;
        return true;
    }
    if (it == gt_poses.begin()) {
        pos_gt = gt_poses.front().second.first;
        quat_gt = gt_poses.front().second.second;
        return true;
    }

    auto it_prev = std::prev(it);
    const int64_t t1 = it_prev->first;
    const int64_t t2 = it->first;
    if (t2 == t1) {
        pos_gt = it->second.first;
        quat_gt = it->second.second;
        return true;
    }

    const double alpha = static_cast<double>(t_ns - t1) / static_cast<double>(t2 - t1);
    pos_gt = (1.0 - alpha) * it_prev->second.first + alpha * it->second.first;
    quat_gt = it_prev->second.second.slerp(alpha, it->second.second);
    return true;
}

class GEORIO
{

public:
    // offline replay: no subscriptions are created
    GEORIO(rclcpp::Node::SharedPtr& nh, bool offline = false)
    {
        offline_mode = offline;
        readParameters(nh);
        if (!if_lidar_only) {
            std::string imu_type = CommonUtils::readParam<std::string>(nh, "topic_imu");
            imu_topic_ = imu_type;
            if (!offline_mode) {
                sub_imu = nh->create_subscription<sensor_msgs::msg::Imu>(imu_type, 2000000, std::bind(&GEORIO::getImuCallback, this, std::placeholders::_1));
            }
        }        
        pub_est = nh->create_publisher<estimate_msgs::msg::Estimate>("est_window", 50);
        pub_start_time = nh->create_publisher<std_msgs::msg::Int64>("start_time", 50);
        pub_cur_scan = nh->create_publisher<sensor_msgs::msg::PointCloud2>("current_scan", 2);
        br = std::make_shared<tf2_ros::TransformBroadcaster>(nh);        
        auto lidar_names = nh->declare_parameter<std::vector<std::string>>("lidars", std::vector<std::string>());
        assert(nh->get_parameter({"lidars"}, lidar_names));
        if (lidar_names.empty()) {
            const std::string lidar_prefix = "";
            LidarConfig lidar(nh, lidar_prefix);
            const std::string lidar_name = lidar.type;
            lidars.emplace(lidar_name, lidar);
            lidars_data.emplace(std::piecewise_construct, std::make_tuple(lidar_name), std::make_tuple());
            radar_ransac_configs[lidar_name] = loadRadarRansacConfig(nh, lidar_prefix, lidar.type);
        } else {
            for (const auto& lidar_name : lidar_names) {
                const std::string lidar_prefix = lidar_name + ".";
                LidarConfig lidar(nh, lidar_prefix);
                lidars.emplace(lidar_name, lidar);
                lidars_data.emplace(std::piecewise_construct, std::make_tuple(lidar_name), std::make_tuple());
                radar_ransac_configs[lidar_name] = loadRadarRansacConfig(nh, lidar_prefix, lidar.type);
            }
        }    
        if (!offline_mode) {
        for (const auto& [lidar_name, lidar] : lidars) {
            if (!lidar.type.compare("Ouster")) {
                auto sub = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000,
                        [this, lidar_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                            this->ousterLidarCallback<ouster_ros::Point>(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("Mid70Avia")) {
                auto sub = nh->create_subscription<livox_ros_driver::msg::CustomMsg>(
                        lidar.topic, 200000,
                        [this, lidar_name](const livox_ros_driver::msg::CustomMsg::SharedPtr msg) {
                            this->livoxLidarCallback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("HAP360")) {
                auto sub = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(
                        lidar.topic, 200000,
                        [this, lidar_name](const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
                            this->livoxLidar2Callback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("AviaResple")) {
                auto sub = nh->create_subscription<livox_interfaces::msg::CustomMsg>(
                        lidar.topic, 200000,
                        [this, lidar_name](const livox_interfaces::msg::CustomMsg::SharedPtr msg) {
                            this->livoxAVIACallback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("Hesai")) {
                auto sub = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000,
                        [this, lidar_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                            this->hesaiLidarCallback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("Mid360Boxi")) {
                auto sub = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000,
                        [this, lidar_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                            this->livoxMid360BoxiCallback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("SnailContinental")) {
                auto sub = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000,
                        [this, lidar_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                            this->snailContiCallback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("Hugin")) {
                auto sub = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000,
                        [this, lidar_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                            this->huginCallback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("HkustContinental")) {
                auto sub = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000,
                        [this, lidar_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                            this->hkustContiCallback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            } else if (!lidar.type.compare("HerculesContinental")) {
                auto sub = nh->create_subscription<sensor_msgs::msg::PointCloud2>(
                        lidar.topic, 200000,
                        [this, lidar_name](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                            this->herculesContiCallback(msg, lidar_name);
                        });
                lidar_subscriptions.push_back(std::static_pointer_cast<rclcpp::SubscriptionBase>(sub));
            }
        }
        }

        VoxelConfig voxel_config(nh);
        voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));
        voxelmap_manager->voxel_map_pub_ = nh->create_publisher<visualization_msgs::msg::MarkerArray>("voxel_map", 10000);

        result_file_name_sec = CommonUtils::readParam<std::string>(nh, "result_file_path", "georio_result.txt");
        result_file_name_nsec = CommonUtils::readParam<std::string>(nh, "result_file_path_ns", "georio_result_ns.txt");
        unc_file_name = CommonUtils::readParam<std::string>(nh, "unc_file_path", "georio_unc.txt");

        std::ofstream clear_file(result_file_name_sec,std::ios::out | std::ios::trunc);
        std::ofstream clear_file_nsec(result_file_name_nsec,std::ios::out | std::ios::trunc);
        std::ofstream clear_file_unc(unc_file_name,std::ios::out | std::ios::trunc);

        pc_map_tot.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        pc_map_gtpose.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        pub_kdtree_map = nh->create_publisher<sensor_msgs::msg::PointCloud2>("kdtree_map", 10);
        pub_odom = nh->create_publisher<nav_msgs::msg::Odometry>("odometry_georio", 500);
        pub_gt_odom = nh->create_publisher<nav_msgs::msg::Odometry>("/gt_odometry", 500);

        if (!offline_mode) {
            sub_command = nh->create_subscription<std_msgs::msg::String>(
                "/command", 10, std::bind(&GEORIO::command_callback, this, std::placeholders::_1));
        }

        const std::string gt_path = CommonUtils::readParam<std::string>(nh, "gt_path", "");
        const std::string gt_time_unit = CommonUtils::readParam<std::string>(nh, "gt_time_unit", "s");
        if (!gt_path.empty()) {
            try {
                gt_pose = load_tum_as_ns(gt_path, gt_time_unit);
            } catch (const std::exception& e) {
                RCLCPP_WARN(
                    nh->get_logger(),
                    "Failed to load gt_path '%s' (%s). Continue without GT pose.",
                    gt_path.c_str(),
                    e.what());
                gt_pose.clear();
            }
        } else {
            RCLCPP_WARN(nh->get_logger(), "gt_path is empty. Continue without GT pose.");
            gt_pose.clear();
        }
    }

    void processData()
    {
        RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "Starting GEORIO processing...");
        rclcpp::Rate rate(20);
        while (rclcpp::ok()) {
            if (!processStep()) {
                rate.sleep();
            }
        }
    }

    // One pass over the buffered data; returns whether any update ran.
    bool processStep()
    {
        int64_t& max_spl_knots = max_spl_knots_;
        int64_t& t_last_map_upd = t_last_map_upd_;
        {
            for (auto& [lidar_name, lidar_data] : lidars_data) {
                while (!lidar_data.t_buff.empty()) {
                    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_frame(new pcl::PointCloud<pcl::PointXYZINormal>());
                    lidar_data.mtx_pc.lock();
                    pc_frame->points = lidar_data.pc_buff.front();
                    lidar_data.pc_buff.pop_front();
                    int64_t time_begin = lidar_data.t_buff.front();
                    lidar_data.t_buff.pop_front();
                    lidar_data.mtx_pc.unlock();
                    std::vector<int> indices;
                    pcl::removeNaNFromPointCloud(*pc_frame, *pc_frame, indices);
                    pc_last_ds->clear();

                    // downsample
                    size_t downsample_step = 1;  // 간격
                    for (size_t i = 0; i < pc_frame->points.size(); i += downsample_step) {
                        pc_last_ds->push_back(pc_frame->points[i]);
                    }

                    sort(pc_last_ds->points.begin(), pc_last_ds->points.end(), &CommonUtils::time_list);
                    const LidarConfig& lidar = lidars.at(lidar_name);
                    for (size_t i = 0; i < pc_last_ds->points.size(); i++) {
                        PointData pt(pc_last_ds->points[i], time_begin, lidar.q_bl, lidar.t_bl, lidar.w_pt, lidar.w_vel);
                        lidar_data.pt_buff.push_back(pt);
                    }
                }
            }            
            if (!if_lidar_only && !imu_int_buff.empty()) {
                m_buff.lock();
                Eigen::aligned_vector<sensor_msgs::msg::Imu::SharedPtr> imu_buff_msg = imu_int_buff;
                imu_int_buff.clear();
                m_buff.unlock();
                for (size_t i = 0; i < imu_buff_msg.size(); i++) {
                    const auto imu_msg = imu_buff_msg[i];
                    int64_t t_ns = rclcpp::Time(imu_msg->header.stamp).nanoseconds();
                    Eigen::Vector3d acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                    if (acc_ratio) acc *= 9.81;
                    Eigen::Vector3d gyro(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                    ImuData imu(t_ns, gyro, acc); 
                    imu_buff.push_back(imu);
                }
            }
            if(!initialization()) {
                return false;
            }
            bool did_update = false;
            while (collectMeasurements()) {
                did_update = true;
                int64_t max_time_ns = pt_meas.back().time_ns;
                if (if_lidar_only) {
                    estimator_lo.propRCP(max_time_ns);
                    estimator_lo.updateIEKFLiDAR(pt_meas, &ikdtree, param.nn_thresh, param.coeff_cov);  
                } else {
                    if (!imu_meas.empty()) {
                        max_time_ns = std::max(imu_meas.back().time_ns, max_time_ns);
                    }
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "=====Aft collection=====");
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "spline max time: %.9f", double(spline->maxTimeNs()) * 1e-9);
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "spline min time: %.9f", double(spline->maxTimeNs() - spline->getKnotTimeIntervalNs()) * 1e-9);
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "imu max time   : %.9f", double(imu_meas.back().time_ns) * 1e-9);
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "imu min time   : %.9f", double(imu_meas.front().time_ns) * 1e-9);
                    
                    while (!imu_meas.empty() && imu_meas.front().time_ns < spline->maxTimeNs() - spline->getKnotTimeIntervalNs()) {
                        imu_meas.pop_front();
                    }
                    // auto T1 = std::chrono::steady_clock::now();
                    estimator_lio.propRCP(max_time_ns);
                    // auto T2 = std::chrono::steady_clock::now();
                    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(T2 - T1).count();

                    // std::ofstream time_file("time.txt", std::ios::app);
                    // time_file.setf(std::ios::fixed, std::ios::floatfield);
                    // time_file.precision(9);
                    // time_file << pt_meas.back().time_ns << "," << duration << " ";
                    // time_file.close();


                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "=====Estimation=====");
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "spline max time: %.9f", double(spline->maxTimeNs()) * 1e-9);
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "spline min time: %.9f", double(spline->maxTimeNs() - spline->getKnotTimeIntervalNs()) * 1e-9);
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "imu max time   : %.9f", double(imu_meas.back().time_ns) * 1e-9);
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "imu min time   : %.9f", double(imu_meas.front().time_ns) * 1e-9);
                    

                    // count the each pt meas size
                    int64_t pt_time = pt_meas.front().time_ns;
                    int pt_cnt = 0;
                    for (auto & pt_data : pt_meas) {
                        if (pt_time != pt_data.time_ns) {
                            // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "pt meas time   : %.9f, count: %d", double(pt_time) * 1e-9, pt_cnt);
                            pt_time = pt_data.time_ns;
                            pt_cnt = 1;
                        }
                        else {
                            pt_cnt++;
                        }
                    }
                    // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "pt meas time   : %.9f, count: %d", double(pt_time) * 1e-9, pt_cnt);

                    // kdtree
                    estimator_lio.updateIEKFLiDARInertial(pt_meas, &ikdtree, param.nn_thresh, imu_meas, gravity, param.cov_acc, param.cov_gyro, param.coeff_cov, param.cov_grav);
                    // voxelmap
                    // estimator_lio.updateIEKFLiDARInertial(pt_meas, voxelmap_manager, param.nn_thresh, imu_meas, gravity, param.cov_acc, param.cov_gyro, param.coeff_cov, param.cov_grav);
                    
                }

                // auto T4 = std::chrono::steady_clock::now();
                #pragma omp parallel for num_threads(NUM_OF_THREAD)
                for (size_t i = 0; i < pt_meas.size(); i++) {
                    PointData& pt_data = pt_meas[i];            
                    // Association::pointBodyToWorld(pt_data.time_ns, spline, pt_data.pt, pt_data.pt_w, pt_data.t_bl, pt_data.q_bl);
                    // compute the pt_w & cov_w for mapping
                    Association::pointLocalToWorld(pt_data.time_ns, spline, pt_data, pt_data);
                }
                for (size_t i = 0; i < pt_meas.size(); i++) {
                    PointData& pt_data = pt_meas[i];
                    pt_data.pt_w.normal_y = pt_data.is_plane ? 1.0 : 0.0; // store plane flag in normal_y
                    pc_world.points.push_back(pt_data.pt_w);
                    accum_nearest_points.push_back(pt_data.nearest_points);

                    // pt_data_buff.push_back(pt_data);
                }
                


                // #pragma omp parallel for num_threads(NUM_OF_THREAD)
                // for (size_t i = 0; i < pt_meas.size(); i++) {
                //     PointData& pt_data = pt_meas[i];
                //     int64_t t_ns = pt_data.time_ns;        
                //     // find the gt pose using iterator
                //     auto it = std::lower_bound(gt_pose.begin(), gt_pose.end(), t_ns, [](const TumEntry& entry, int64_t t) {
                //         return entry.first < t;
                //     });
                //     Eigen::Vector3d pos_gt;
                //     Eigen::Quaterniond quat_gt;
                //     if (it == gt_pose.end()) {
                //         pos_gt = (gt_pose.end()-1)->second.first;
                //         quat_gt = (gt_pose.end()-1)->second.second;
                //     } else if (it == gt_pose.begin()) {
                //         pos_gt = gt_pose.begin()->second.first;
                //         quat_gt = gt_pose.begin()->second.second;                
                //     } else {
                //         auto it_prev = std::prev(it);
                //         int64_t t1 = it_prev->first;
                //         int64_t t2 = it->first;
                //         double alpha = double(t_ns - t1) / double(t2 - t1);
                //         // RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "t_ns: %ld, t1: %ld, t2: %ld, alpha: %.9f", t_ns, t1, t2, alpha);
                //         pos_gt = (1.0 - alpha) * it_prev->second.first + alpha * it->second.first;
                //         quat_gt = it_prev->second.second.slerp(alpha, it->second.second);
                //     }

                //     PointData pt_data_gt;
                //     Eigen::Vector3d p_lid(pt_data.pt.x, pt_data.pt.y, pt_data.pt.z);
                //     Eigen::Vector3d p_glb = pos_gt + quat_gt * (p_lid);
                //     pt_data_gt.pt_w.x = p_glb.x();
                //     pt_data_gt.pt_w.y = p_glb.y();
                //     pt_data_gt.pt_w.z = p_glb.z();
                //     pt_data_gt.pt_w.intensity = pt_data.pt_w.intensity;
                //     pt_data_gt.pt_w.curvature = pt_data.pt_w.curvature;
                //     pt_data_gt.pt_w.normal_x = pt_data.pt_w.normal_x;
                //     pt_data_gt.pt_w.normal_y = pt_data.is_plane ? 1.0 : 0.0; // store plane flag in normal_y
                //     pc_world_gtpose.points.push_back(pt_data_gt.pt_w);
                // }

                pt_meas.clear();


                // ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                // pc_map_tot->clear();
                // pc_map_tot->points = ikdtree.PCL_Storage;

                // if (pc_map_tot->points.size() > 100000) {
                //     for (size_t i = pc_map_tot->points.size(); i-- > 0; ) {
                //         if (i % 5 != 0) {
                //             pc_map_tot->points.erase(pc_map_tot->points.begin() + i);
                //         }
                //     }
                //     pc_map_tot->width = pc_map_tot->points.size();
                //     pc_map_tot->height = 1;
                // }

                sensor_msgs::msg::PointCloud2 mapCloudmsg;
                pcl::toROSMsg(*pc_map_tot, mapCloudmsg);
                mapCloudmsg.header.stamp = rclcpp::Time(spline->maxTimeNs());
                mapCloudmsg.header.frame_id = odom_id;
                pub_kdtree_map->publish(mapCloudmsg);

                if (spline->numKnots() > max_spl_knots) {
                    estimate_msgs::msg::Spline spline_msg;
                    spline->getSplineMsg(spline_msg, std::max(int(max_spl_knots-1),0));
                    estimate_msgs::msg::Estimate est_msg;
                    est_msg.spline = spline_msg;
                    est_msg.if_full_window.data = (spline->numKnots() >= 4);
                    est_msg.runtime.data = 0;
                    pub_est->publish(est_msg);
                    max_spl_knots = spline->numKnots();
                }
                if (max_time_ns >= t_last_map_upd + dt_ns) {
                    mapIncremental();
                    publishFrameWorld();
                    lasermapFovSegment();
                    


                    // voxelmap_manager->pt_data_buff_.swap(pt_data_buff);
                    // voxelmap_manager->updateVoxelMap(voxelmap_manager->pt_data_buff_);
                    // if(voxelmap_manager->config_setting_.map_sliding_en)
                    // {
                    //     voxelmap_manager->mapSliding(max_time_ns, spline);
                    // }
                    
                    // if(voxelmap_manager->config_setting_.pub_voxelmap_en_)
                    // {
                    //     voxelmap_manager->pubVoxelMap();
                    // }

                    
                    Eigen::Matrix3d pos_unc, rot_unc;
                    Eigen::Vector3d pos = spline->itpPosition(max_time_ns, nullptr, &pos_unc);
                    Eigen::Quaterniond orient;
                    spline->itpQuaternion(max_time_ns, &orient, nullptr, nullptr, nullptr, &rot_unc);
                    std::ofstream result_file(result_file_name_sec.c_str(), std::ios::app);
                    result_file.setf(std::ios::fixed, std::ios::floatfield);
                    result_file.precision(9);
                    int64_t sec = max_time_ns / 1'000'000'000;
                    int64_t nsec = max_time_ns % 1'000'000'000;

                    result_file << sec << '.'
                                << std::setw(9) << std::setfill('0') << nsec << ' '
                                << pos.x() << ' ' << pos.y() << ' ' << pos.z() << ' '
                                << orient.x() << ' ' << orient.y() << ' ' << orient.z() << ' ' << orient.w()
                                << '\n';
                    result_file.close();

                    std::ofstream result_file_ns(result_file_name_nsec.c_str(), std::ios::app);
                    result_file_ns.setf(std::ios::fixed, std::ios::floatfield);
                    result_file_ns.precision(9);

                    result_file_ns << max_time_ns << ' '
                                << pos.x() << ' ' << pos.y() << ' ' << pos.z() << ' '
                                << orient.x() << ' ' << orient.y() << ' ' << orient.z() << ' ' << orient.w()
                                << '\n';
                    result_file_ns.close();

                    double pos_unc_ = pos_unc(0, 0) + pos_unc(1, 1) + pos_unc(2, 2);
                    double rot_unc_ = rot_unc(0, 0) + rot_unc(1, 1) + rot_unc(2, 2);

                    std::ofstream unc_file(unc_file_name.c_str(), std::ios::app);
                    unc_file.setf(std::ios::fixed, std::ios::floatfield);
                    unc_file.precision(9);

                    unc_file << max_time_ns << ' '
                             << pos_unc_ * 100 << ' ' << rot_unc_ * 100 << ' ' << (pos_unc_ + rot_unc_) * 100 << '\n';
                    unc_file.close();



                    pc_world.clear();
                    // pc_world_gtpose.clear();
                    accum_nearest_points.clear();
                    t_last_map_upd = max_time_ns;

                    nav_msgs::msg::Odometry odom_msg;
                    odom_msg.header.stamp = rclcpp::Time(max_time_ns);
                    odom_msg.header.frame_id = odom_id;
                    odom_msg.child_frame_id = frame_id;
                    odom_msg.pose.pose.position.x = pos.x();
                    odom_msg.pose.pose.position.y = pos.y();
                    odom_msg.pose.pose.position.z = pos.z();
                    odom_msg.pose.pose.orientation.x = orient.x();
                    odom_msg.pose.pose.orientation.y = orient.y();
                    odom_msg.pose.pose.orientation.z = orient.z();
                    odom_msg.pose.pose.orientation.w = orient.w();
                    pub_odom->publish(odom_msg);

                    Eigen::Vector3d pos_gt;
                    Eigen::Quaterniond quat_gt;
                    if (interpolate_gt_pose_ns(gt_pose, max_time_ns, pos_gt, quat_gt)) {
                        nav_msgs::msg::Odometry gt_odom_msg;
                        gt_odom_msg.header.stamp = rclcpp::Time(max_time_ns);
                        gt_odom_msg.header.frame_id = odom_id;
                        gt_odom_msg.child_frame_id = "gt_base_link";
                        gt_odom_msg.pose.pose.position.x = pos_gt.x();
                        gt_odom_msg.pose.pose.position.y = pos_gt.y();
                        gt_odom_msg.pose.pose.position.z = pos_gt.z();
                        gt_odom_msg.pose.pose.orientation.x = quat_gt.x();
                        gt_odom_msg.pose.pose.orientation.y = quat_gt.y();
                        gt_odom_msg.pose.pose.orientation.z = quat_gt.z();
                        gt_odom_msg.pose.pose.orientation.w = quat_gt.w();
                        pub_gt_odom->publish(gt_odom_msg);
                    }
                }
            }
            return did_update;
        }
    }

    void setupOffline()
    {
        offline_mode = true;
        ikd_skip_large_rebuild = true;
    }

    std::string herculesRadarName() const
    {
        for (const auto& [name, lidar] : lidars) {
            if (lidar.type.compare("HerculesContinental") == 0) return name;
        }
        return "";
    }

    // Replay a HeRCULES sequence folder (raw_data/<Place>/<run>) in data_stamp.csv order.
    void runOfflineHercules(const std::string& seq_dir)
    {
        setupOffline();
        namespace fs = std::filesystem;
        const std::string radar_name = herculesRadarName();
        if (radar_name.empty()) {
            RCLCPP_ERROR(rclcpp::get_logger("GEORIO"), "offline hercules: no lidar of type HerculesContinental in the config");
            return;
        }
        fs::path radar_dir;
        for (const char* cand : {"radar/continental", "Radar/Continental"}) {
            if (fs::is_directory(fs::path(seq_dir) / cand)) { radar_dir = fs::path(seq_dir) / cand; break; }
        }
        if (radar_dir.empty()) {
            RCLCPP_ERROR(rclcpp::get_logger("GEORIO"), "offline hercules: no radar/continental folder under %s", seq_dir.c_str());
            return;
        }
        // IMU rows keyed by stamp: stamp,qx,qy,qz,qw,ex,ey,ez,gx,gy,gz,ax,ay,az,mx,my,mz
        std::unordered_map<int64_t, sensor_msgs::msg::Imu> imu_rows;
        {
            std::ifstream f(fs::path(seq_dir) / "sensor_data" / "xsens_imu.csv");
            std::string line;
            while (std::getline(f, line)) {
                std::replace(line.begin(), line.end(), ',', ' ');
                std::istringstream ss(line);
                int64_t stamp; double v[16];
                if (!(ss >> stamp)) continue;
                int n = 0;
                while (n < 16 && (ss >> v[n])) ++n;
                if (n < 13) continue;
                sensor_msgs::msg::Imu m;
                m.header.stamp = rclcpp::Time(stamp);
                m.header.frame_id = "imu";
                m.orientation.x = v[0]; m.orientation.y = v[1]; m.orientation.z = v[2]; m.orientation.w = v[3];
                m.angular_velocity.x = v[7]; m.angular_velocity.y = v[8]; m.angular_velocity.z = v[9];
                m.linear_acceleration.x = v[10]; m.linear_acceleration.y = v[11]; m.linear_acceleration.z = v[12];
                imu_rows.emplace(stamp, std::move(m));
            }
        }
        RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "offline hercules: %zu imu rows, radar dir %s", imu_rows.size(), radar_dir.c_str());

        std::ifstream stamps(fs::path(seq_dir) / "sensor_data" / "data_stamp.csv");
        std::string line;
        size_t n_frames = 0, n_imu = 0, n_missing = 0;
        while (rclcpp::ok() && std::getline(stamps, line)) {
            const auto comma = line.find(',');
            if (comma == std::string::npos) continue;
            const int64_t stamp = std::stoll(line.substr(0, comma));
            std::string type = line.substr(comma + 1);
            while (!type.empty() && (type.back() == '\r' || type.back() == ' ')) type.pop_back();
            if (type == "imu") {
                auto it = imu_rows.find(stamp);
                if (it == imu_rows.end()) continue;
                getImuCallback(std::make_shared<sensor_msgs::msg::Imu>(it->second));
                ++n_imu;
            } else if (type == "continental") {
                // packed records: x y z v r (f32) | RCS (i8) | azimuth elevation (f32) = 29 bytes
                std::ifstream bin(radar_dir / (std::to_string(stamp) + ".bin"), std::ios::binary);
                if (!bin) { ++n_missing; continue; }
                pcl::PointCloud<hercules_conti::Point> cloud;
                char rec[29];
                while (bin.read(rec, sizeof(rec))) {
                    hercules_conti::Point pt;
                    float f[5]; std::memcpy(f, rec, 20);
                    pt.x = f[0]; pt.y = f[1]; pt.z = f[2]; pt.v = f[3]; pt.r = f[4];
                    pt.RCS = static_cast<std::int8_t>(rec[20]);
                    float ae[2]; std::memcpy(ae, rec + 21, 8);
                    pt.azimuth = ae[0]; pt.elevation = ae[1];
                    pt.intensity = 0.f;
                    cloud.points.push_back(pt);
                }
                cloud.width = cloud.points.size(); cloud.height = 1; cloud.is_dense = true;
                auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
                pcl::toROSMsg(cloud, *msg);
                msg->header.stamp = rclcpp::Time(stamp);
                msg->header.frame_id = "continental";
                herculesContiCallback(msg, radar_name);
                ++n_frames;
                if (offline_frames_per_step <= 1 || n_frames % offline_frames_per_step == 0) {
                    while (processStep()) {}
                }
                if (n_frames % 500 == 0) {
                    RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "offline hercules: %zu radar frames, %zu imu, %ld knots", n_frames, n_imu, (long)spline->numKnots());
                }
            }
        }
        while (processStep()) {}
        RCLCPP_WARN(rclcpp::get_logger("GEORIO"), "offline hercules done: %zu radar frames (%zu missing files), %zu imu messages", n_frames, n_missing, n_imu);
    }

    // Replay a ROS 2 bag, stepping the estimator every N radar frames.
    void runOffline(const std::string& bag_uri)
    {
        setupOffline();
        std::map<std::string, std::pair<std::string, std::string>> radar_topic_to_name;  // topic -> (name, type)
        for (const auto& [name, lidar] : lidars) {
            if (lidar.type.compare("SnailContinental") == 0 ||
                lidar.type.compare("HkustContinental") == 0) {
                radar_topic_to_name[lidar.topic] = {name, lidar.type};
            } else {
                RCLCPP_WARN(rclcpp::get_logger("GEORIO"), "offline mode: lidar type %s is not supported, %s ignored", lidar.type.c_str(), name.c_str());
            }
        }
        rosbag2_cpp::Reader reader;
        reader.open(bag_uri);
        rosbag2_storage::StorageFilter filter;
        for (const auto& [topic, entry] : radar_topic_to_name) filter.topics.push_back(topic);
        if (!if_lidar_only) filter.topics.push_back(imu_topic_);
        reader.set_filter(filter);
        rclcpp::Serialization<sensor_msgs::msg::Imu> imu_ser;
        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> pc_ser;
        size_t n_frames = 0, n_imu = 0;
        while (rclcpp::ok() && reader.has_next()) {
            auto bag_msg = reader.read_next();
            rclcpp::SerializedMessage smsg(*bag_msg->serialized_data);
            if (!if_lidar_only && bag_msg->topic_name == imu_topic_) {
                auto msg = std::make_shared<sensor_msgs::msg::Imu>();
                imu_ser.deserialize_message(&smsg, msg.get());
                getImuCallback(msg);
                ++n_imu;
                continue;
            }
            auto it = radar_topic_to_name.find(bag_msg->topic_name);
            if (it == radar_topic_to_name.end()) continue;
            auto msg = std::make_shared<sensor_msgs::msg::PointCloud2>();
            pc_ser.deserialize_message(&smsg, msg.get());
            if (it->second.second.compare("HkustContinental") == 0) {
                hkustContiCallback(msg, it->second.first);
            } else {
                snailContiCallback(msg, it->second.first);
            }
            ++n_frames;
            if (offline_frames_per_step <= 1 || n_frames % offline_frames_per_step == 0) {
                while (processStep()) {}
            }
            if (n_frames % 500 == 0) {
                RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "offline: %zu radar frames, %zu imu messages, %ld knots", n_frames, n_imu, (long)spline->numKnots());
            }
        }
        while (processStep()) {}
        RCLCPP_WARN(rclcpp::get_logger("GEORIO"), "offline replay done: %zu radar frames, %zu imu messages", n_frames, n_imu);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

    std::string node_name = "GEORIO";
    std::vector<rclcpp::SubscriptionBase::SharedPtr> lidar_subscriptions;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_command;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cur_scan;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_kdtree_map;
    rclcpp::Publisher<estimate_msgs::msg::Estimate>::SharedPtr pub_est;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_gt_odom;
    rclcpp::Publisher<std_msgs::msg::Int64>::SharedPtr pub_start_time;
    std::shared_ptr<tf2_ros::TransformBroadcaster> br;
    const std::string frame_id = "base_link";
    const std::string odom_id = "odom";    

    std::map<std::string, LidarConfig> lidars;
    float ds_lm_voxel;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last;
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last_ds;
    pcl::PointCloud<pcl::PointXYZINormal> pc_world;
    pcl::PointCloud<pcl::PointXYZINormal> pc_world_gtpose;

    Eigen::aligned_deque<PointData> pt_data_buff;

    int point_filter_num = 1;
    int64_t time_offset = 0;

    Eigen::Vector3d prev_ego_vel = Eigen::Vector3d::Zero();


    struct RadarRansacConfig {
        int sample_size = 5;
        double outlier_prob = 0.05;
        double success_prob = 0.995;
        double inlier_threshold = 0.1;
        std::size_t min_static_points = 30;
        std::string mode = "3d";
        double max_vel_jump = 2.0;   // reject an ego-velocity this far from the previous one
        double max_vel_z = 0.3;      // reject an ego-velocity with a larger vertical component
    };

    struct RadarRansacResult {
        bool success = false;
        Eigen::Vector3d ego_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d sigma_v_r = Eigen::Vector3d::Zero();
        std::size_t raw_point_count = 0;
        std::size_t static_point_count = 0;
    };

    std::unordered_map<std::string, RadarRansacConfig> radar_ransac_configs;
    std::unordered_map<std::string, int64_t> lidar_last_t_ns;

    int64_t getLastLidarTimestampNs(const std::string& lidar_name, int64_t default_t_ns) const
    {
        const auto it = lidar_last_t_ns.find(lidar_name);
        if (it == lidar_last_t_ns.end()) {
            return default_t_ns;
        }
        return it->second;
    }

    void setLastLidarTimestampNs(const std::string& lidar_name, int64_t t_ns)
    {
        lidar_last_t_ns[lidar_name] = t_ns;
    }

    RadarRansacConfig defaultRadarRansacConfig(const std::string& lidar_type) const
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

    RadarRansacConfig loadRadarRansacConfig(rclcpp::Node::SharedPtr& nh,
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
        config.max_vel_jump = CommonUtils::readParam<double>(nh, ransac_prefix + "max_vel_jump", config.max_vel_jump);
        config.max_vel_z = CommonUtils::readParam<double>(nh, ransac_prefix + "max_vel_z", config.max_vel_z);
        return config;
    }

    double computeRadarNormalizationDenominator(const pcl::PointXYZINormal& radar_pt,
                                                const RadarRansacConfig& config) const
    {
        const bool use_3d_ransac = config.mode != "2d" && config.mode != "2D";
        double range_sq = radar_pt.x * radar_pt.x + radar_pt.y * radar_pt.y;
        if (use_3d_ransac) {
            range_sq += radar_pt.z * radar_pt.z;
        }
        const double range_norm = std::sqrt(range_sq);
        return range_norm > 1.0e-9 ? range_norm : 1.0e-9;
    }

    int computeRadarRansacIterations(const RadarRansacConfig& config) const
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

    std::vector<uint> collectRadarInliers(const Eigen::MatrixXd& H_all,
                                          const Eigen::VectorXd& y_all,
                                          const Eigen::Vector3d& velocity,
                                          double inlier_threshold) const
    {
        const Eigen::VectorXd err = (y_all - H_all * velocity).array().abs();
        std::vector<uint> inlier_idx;
        inlier_idx.reserve(err.rows());
        for (int j = 0; j < err.rows(); ++j) {
            if (err(j) < inlier_threshold) {
                inlier_idx.emplace_back(static_cast<uint>(j));
            }
        }
        return inlier_idx;
    }

    void fillRadarStaticScan(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& source_scan,
                             const std::vector<uint>& inlier_idx,
                             const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& static_scan) const
    {
        static_scan->points.clear();
        static_scan->points.reserve(inlier_idx.size());
        for (const uint idx : inlier_idx) {
            static_scan->points.push_back(source_scan->points[idx]);
        }
    }

    void logRadarVelocityFallback(const Eigen::Vector3d& ego_vel,
                                  const Eigen::Vector3d& sigma_v_r) const
    {
        RCLCPP_WARN(rclcpp::get_logger("radar"), "Prev ego vel: %f, %f, %f", -prev_ego_vel.x(), -prev_ego_vel.y(), -prev_ego_vel.z());
        RCLCPP_WARN(rclcpp::get_logger("radar"), "Curr Ego vel: %f, %f, %f", -ego_vel.x(), -ego_vel.y(), -ego_vel.z());
        RCLCPP_WARN(rclcpp::get_logger("radar"), "Sigma vel   : %f, %f, %f", sigma_v_r.x(), sigma_v_r.y(), sigma_v_r.z());
        RCLCPP_WARN(rclcpp::get_logger("radar"), "Wrong ego velocity : use prev egovel");
    }

    RadarRansacResult filterRadarStaticPoints(pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_last,
                                              const RadarRansacConfig& config)
    {
        RadarRansacResult result;
        result.raw_point_count = pc_last->points.size();

        pcl::PointCloud<pcl::PointXYZINormal>::Ptr static_scan(new pcl::PointCloud<pcl::PointXYZINormal>());
        const int filter_size = static_cast<int>(pc_last->points.size());
        Eigen::MatrixXd radar_data(filter_size, 4);
        for (int i = 0; i < filter_size; ++i) {
            const auto& radar_pt = pc_last->points[i];
            const double denom = computeRadarNormalizationDenominator(radar_pt, config);
            radar_data(i, 0) = radar_pt.x / denom;
            radar_data(i, 1) = radar_pt.y / denom;
            radar_data(i, 2) = radar_pt.z / denom;
            radar_data(i, 3) = radar_pt.normal_z;
        }

        if (radar_data.rows() < config.sample_size) {
            std::cerr << "Error: Not enough radar data points for RANSAC" << std::endl;
            return result;
        }

        const int ransac_iter = computeRadarRansacIterations(config);
        std::vector<uint> sample_idx(radar_data.rows());
        for (int k = 0; k < radar_data.rows(); ++k) {
            sample_idx[k] = static_cast<uint>(k);
        }

        static thread_local std::mt19937 generator(20260101u);
        Eigen::MatrixXd H_all(radar_data.rows(), 3);
        H_all.col(0) = radar_data.col(0);
        H_all.col(1) = radar_data.col(1);
        H_all.col(2) = radar_data.col(2);
        const Eigen::VectorXd y_all = radar_data.col(3);

        std::vector<uint> inlier_idx_best;
        for (int k = 0; k < ransac_iter; ++k) {
            std::shuffle(sample_idx.begin(), sample_idx.end(), generator);

            Eigen::MatrixXd radar_data_iter(config.sample_size, 4);
            for (int i = 0; i < config.sample_size; ++i) {
                radar_data_iter.row(i) = radar_data.row(sample_idx.at(i));
            }

            Eigen::MatrixXd H(radar_data_iter.rows(), 3);
            H.col(0) = radar_data_iter.col(0);
            H.col(1) = radar_data_iter.col(1);
            H.col(2) = radar_data_iter.col(2);
            const Eigen::VectorXd y = radar_data_iter.col(3);

            const Eigen::Vector3d candidate_vel =
                H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
            const auto inlier_idx = collectRadarInliers(H_all, y_all, candidate_vel, config.inlier_threshold);
            if (inlier_idx.size() > inlier_idx_best.size()) {
                inlier_idx_best = inlier_idx;
            }
        }

        if (!inlier_idx_best.empty()) {
            Eigen::MatrixXd radar_data_inlier(inlier_idx_best.size(), 4);
            for (uint i = 0; i < inlier_idx_best.size(); ++i) {
                radar_data_inlier.row(i) = radar_data.row(inlier_idx_best.at(i));
            }

            Eigen::MatrixXd H(radar_data_inlier.rows(), 3);
            H.col(0) = radar_data_inlier.col(0);
            H.col(1) = radar_data_inlier.col(1);
            H.col(2) = radar_data_inlier.col(2);
            const Eigen::MatrixXd HTH = H.transpose() * H;
            const Eigen::VectorXd y = radar_data_inlier.col(3);

            result.ego_vel = H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);

            const Eigen::VectorXd e = H * result.ego_vel - y;
            const double dof = static_cast<double>(H.rows() - 3);
            if (dof > 0.0) {
                const Eigen::MatrixXd P_v_r = (e.transpose() * e).x() * HTH.inverse() / dof;
                result.sigma_v_r = Eigen::Vector3d(P_v_r(0, 0), P_v_r(1, 1), P_v_r(2, 2));
            } else {
                result.sigma_v_r = Eigen::Vector3d::Constant(-1.0);
            }

            if (result.sigma_v_r.x() >= 0.0 && result.sigma_v_r.y() >= 0.0 && result.sigma_v_r.z() >= 0.0) {
                result.sigma_v_r = result.sigma_v_r.array().sqrt();
                const bool is_consistent_with_prev =
                    (prev_ego_vel.norm() > 0.1 &&
                     (result.ego_vel - prev_ego_vel).norm() < config.max_vel_jump &&
                     std::abs(result.ego_vel.z()) < config.max_vel_z) ||
                    result.ego_vel.norm() < 0.5;
                if (is_consistent_with_prev) {
                    fillRadarStaticScan(pc_last, inlier_idx_best, static_scan);
                    prev_ego_vel = result.ego_vel;
                } else {
                    // logRadarVelocityFallback(result.ego_vel, result.sigma_v_r);
                    inlier_idx_best = collectRadarInliers(H_all, y_all, prev_ego_vel, config.inlier_threshold);
                    fillRadarStaticScan(pc_last, inlier_idx_best, static_scan);
                    result.ego_vel = prev_ego_vel;
                }
            } else {
                // logRadarVelocityFallback(result.ego_vel, result.sigma_v_r);
                inlier_idx_best = collectRadarInliers(H_all, y_all, prev_ego_vel, config.inlier_threshold);
                fillRadarStaticScan(pc_last, inlier_idx_best, static_scan);
                result.ego_vel = prev_ego_vel;
            }
        }

        if (static_scan->points.size() < config.min_static_points) {
            static_scan->points = pc_last->points;
            // RCLCPP_WARN(rclcpp::get_logger("radar"),
            //             "Not enough points for RANSAC, using %zu raw points",
            //             static_scan->points.size());
        }

        if (result.ego_vel.norm() < 0.1) {
            result.ego_vel = Eigen::Vector3d::Zero();
            for (auto& point : static_scan->points) {
                point.normal_y = 1.0;
            }
            prev_ego_vel = result.ego_vel;
        } else {
            for (auto& point : static_scan->points) {
                point.normal_y = 0.0;
            }
        }

        pc_last->points = static_scan->points;
        result.static_point_count = pc_last->points.size();
        result.success = true;
        return result;
    }

    std::vector<BoxPointType> cub_needrm;
    BoxPointType LocalMap_Points;
    std::vector<Eigen::aligned_vector<pcl::PointXYZINormal>> accum_nearest_points;
    double cube_len = 2000; 
    const float MOV_THRESHOLD = 1.5f;
    float det_range = 100.0;
    bool if_init_map = false;
    struct LidarData {
        Eigen::aligned_deque<Eigen::aligned_vector<pcl::PointXYZINormal>> pc_buff;
        std::deque<int64_t> t_buff;
        std::mutex mtx_pc;
        Eigen::aligned_deque<PointData> pt_buff;
    };
    std::map<std::string, LidarData> lidars_data;    
    Eigen::aligned_deque<PointData> pt_meas;    

    bool if_lidar_only;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu;
    Eigen::aligned_deque<ImuData> imu_buff;
    Eigen::aligned_deque<ImuData> imu_meas;
    Eigen::aligned_vector<sensor_msgs::msg::Imu::SharedPtr> imu_int_buff;    
    std::mutex m_buff;
    std::string imu_topic_;
    bool offline_mode = false;
    int offline_frames_per_step = 1;
    int64_t max_spl_knots_ = 0;
    int64_t t_last_map_upd_ = 0;
    bool acc_ratio;
    Eigen::Vector3d cov_ba;
    Eigen::Vector3d cov_bg;    
    Eigen::Vector3d gravity;
    double cov_grav;

    Eigen::Quaterniond q_WI = Eigen::Quaterniond::Identity(); 
    
    bool if_init_filter = false;
    Estimator<24> estimator_lo;
    Estimator<30> estimator_lio;
    SplineState* spline;
    double cov_P0 = 0.02;
    double cov_RCP_pos_old = 0.02;
    double cov_RCP_ort_old = 0.02;
    double cov_RCP_pos_new = 0.1;    
    double cov_RCP_ort_new = 0.1;    
    double cov_sys_pos = 0.1;    
    double cov_sys_ort = 0.01;    
    Parameters param;
    int64_t dt_ns;
    int num_points_upd;
    
    const std::string baselink_frame = "base_link";
    const std::string odom_frame = "odom";

    std::string result_file_name_sec;
    std::string result_file_name_nsec;
    std::string unc_file_name;

    void saveMap(){
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        pc_map_tot->clear();
        pc_map_tot->points = ikdtree.PCL_Storage;

        // if (pc_map_tot->points.size() > 100000) {
            for (size_t i = pc_map_tot->points.size(); i-- > 0; ) {
                // if (pc_map_tot->points[i].normal_x * 10.0 > 0.55) {
                //     pc_map_tot->points.erase(pc_map_tot->points.begin() + i);
                //     continue;
                // }
                pc_map_tot->points[i].intensity = pc_map_tot->points[i].normal_x * 10; // downsample intensity
                pc_map_tot->points[i].curvature = pc_map_tot->points[i].normal_y; // store plane flag in curvature
            }
            pc_map_tot->width = pc_map_tot->points.size();
            pc_map_tot->height = 1;
        // }
        // save map in PCD format
        std::string map_file_name = "map.pcd";
        pcl::io::savePCDFileBinary(map_file_name, *pc_map_tot);
        RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "Map saved to %s", map_file_name.c_str());
    }

    void saveMap_GTpose(){
        ikdtree_gt.flatten(ikdtree_gt.Root_Node, ikdtree_gt.PCL_Storage, NOT_RECORD);
        pc_map_gtpose->clear();
        pc_map_gtpose->points = ikdtree_gt.PCL_Storage;

        // if (pc_map_gtpose->points.size() > 100000) {
            for (size_t i = pc_map_gtpose->points.size(); i-- > 0; ) {
                // if (pc_map_tot->points[i].normal_x * 10.0 > 0.55) {
                //     pc_map_tot->points.erase(pc_map_tot->points.begin() + i);
                //     continue;
                // }
                pc_map_gtpose->points[i].intensity = pc_map_gtpose->points[i].normal_x * 10; // downsample intensity
                pc_map_gtpose->points[i].curvature = pc_map_gtpose->points[i].normal_y; // store plane flag in curvature
            }
            pc_map_gtpose->width = pc_map_gtpose->points.size();
            pc_map_gtpose->height = 1;
        // }
        // save map in PCD format
        std::string map_file_name = "map_gtpose_ptunc.pcd";
        pcl::io::savePCDFileBinary(map_file_name, *pc_map_gtpose);
        RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "Map with GT pose saved to %s", map_file_name.c_str());
    }

    void command_callback(const std_msgs::msg::String::SharedPtr str_msg)
    {
        if (str_msg->data == "save_map") {
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "Received 'save_map' command, saving map...");
            saveMap();
            // saveMap_GTpose();
        }
    }

    void readParameters(rclcpp::Node::SharedPtr &nh)
    {
        ds_lm_voxel = CommonUtils::readParam<float>(nh, "ds_lm_voxel");
        param.nn_thresh = CommonUtils::readParam<double>(nh, "nn_thresh");
        if_lidar_only = CommonUtils::readParam<bool>(nh, "if_lidar_only");
        if (!if_lidar_only) {
            acc_ratio = CommonUtils::readParam<bool>(nh, "acc_ratio");
            std::vector<double> bias_acc_var = CommonUtils::readParam<std::vector<double>>(nh, "cov_ba");
            cov_ba << bias_acc_var.at(0), bias_acc_var.at(1), bias_acc_var.at(2);
            std::vector<double> bias_gyro_var = CommonUtils::readParam<std::vector<double>>(nh, "cov_bg");
            cov_bg << bias_gyro_var.at(0), bias_gyro_var.at(1), bias_gyro_var.at(2);
            std::vector<double> acc_var = CommonUtils::readParam<std::vector<double>>(nh, "cov_acc");
            param.cov_acc << acc_var.at(0), acc_var.at(1), acc_var.at(2);
            std::vector<double> gyro_var = CommonUtils::readParam<std::vector<double>>(nh, "cov_gyro");
            param.cov_gyro << gyro_var.at(0), gyro_var.at(1), gyro_var.at(2);

            double grav_var = CommonUtils::readParam<double>(nh, "cov_grav", 0.01);
            param.cov_grav = grav_var;
        }

        dt_ns = 1e9 / CommonUtils::readParam<int>(nh, "knot_hz");        
        double dt_s = double(dt_ns) * 1e-9;
        cov_P0 = CommonUtils::readParam<double>(nh, "cov_P0");
        cov_P0 *= (dt_s*dt_s);
        cov_RCP_pos_old = CommonUtils::readParam<double>(nh, "cov_RCP_pos_old");
        cov_RCP_ort_old = CommonUtils::readParam<double>(nh, "cov_RCP_ort_old");
        cov_RCP_pos_new = CommonUtils::readParam<double>(nh, "cov_RCP_pos_new");
        cov_RCP_ort_new = CommonUtils::readParam<double>(nh, "cov_RCP_ort_new");
        double std_pos = CommonUtils::readParam<double>(nh, "std_sys_pos");
        double std_ort = CommonUtils::readParam<double>(nh, "std_sys_ort");
        cov_sys_pos = std_pos*std_pos*dt_s*dt_s;
        cov_sys_ort = std_ort*std_ort*dt_s*dt_s;
        param.coeff_cov = CommonUtils::readParam<double>(nh, "coeff_cov", 10);

        cube_len = CommonUtils::readParam<double>(nh, "cube_len");
        point_filter_num = CommonUtils::readParam<int>(nh, "point_filter_num");
        num_points_upd = CommonUtils::readParam<int>(nh, "num_points_upd");
        if (if_lidar_only) {
            estimator_lo.n_iter = CommonUtils::readParam<int>(nh, "n_iter");
            estimator_lo.min_cov_thresh = CommonUtils::readParam<double>(nh, "min_cov_thresh", 0.01);
            estimator_lo.max_cov_thresh = CommonUtils::readParam<double>(nh, "max_cov_thresh", 0.1);
            estimator_lo.min_cov_bound = CommonUtils::readParam<double>(nh, "min_cov_bound", 1.0);
            estimator_lo.max_cov_bound = CommonUtils::readParam<double>(nh, "max_cov_bound", 3.0);
            estimator_lo.min_plane_cov_thresh = CommonUtils::readParam<double>(nh, "min_plane_cov_thresh", 0.005);
            estimator_lo.max_plane_cov_thresh = CommonUtils::readParam<double>(nh, "max_plane_cov_thresh", 0.01);
            estimator_lo.min_plane_cov_bound = CommonUtils::readParam<double>(nh, "min_plane_cov_bound", 0.0075);
            estimator_lo.max_plane_cov_bound = CommonUtils::readParam<double>(nh, "max_plane_cov_bound", 0.0125);
        } else {
            estimator_lio.n_iter = CommonUtils::readParam<int>(nh, "n_iter");
            estimator_lio.min_cov_thresh = CommonUtils::readParam<double>(nh, "min_cov_thresh", 0.01);
            estimator_lio.max_cov_thresh = CommonUtils::readParam<double>(nh, "max_cov_thresh", 0.1);
            estimator_lio.min_cov_bound = CommonUtils::readParam<double>(nh, "min_cov_bound", 1.0);
            estimator_lio.max_cov_bound = CommonUtils::readParam<double>(nh, "max_cov_bound", 3.0);
            estimator_lio.min_plane_cov_thresh = CommonUtils::readParam<double>(nh, "min_plane_cov_thresh", 0.005);
            estimator_lio.max_plane_cov_thresh = CommonUtils::readParam<double>(nh, "max_plane_cov_thresh", 0.01);
            estimator_lio.min_plane_cov_bound = CommonUtils::readParam<double>(nh, "min_plane_cov_bound", 0.0075);
            estimator_lio.max_plane_cov_bound = CommonUtils::readParam<double>(nh, "max_plane_cov_bound", 0.0125);
        }
        pc_last.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        pc_last_ds.reset(new pcl::PointCloud<pcl::PointXYZINormal>());
        NUM_MATCH_POINTS = CommonUtils::readParam<int>(nh, "num_nn", 5);
        plane_thresh = CommonUtils::readParam<float>(nh, "plane_thresh", 0.3);
        offline_frames_per_step = CommonUtils::readParam<int>(nh, "offline_frames_per_step", 1);
        double lidar_time_offset = CommonUtils::readParam<double>(nh, "lidar_time_offset", 0.0);
        time_offset = 1e9*lidar_time_offset;

        RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "Parameters read successfully:");
        RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "ds_lm_voxel: %.3f", ds_lm_voxel);
        RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "nn_thresh: %.3f", param.nn_thresh);
        RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "if_lidar_only: %s", if_lidar_only ? "true" : "false");
        
        if (!if_lidar_only) {
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "min_cov_thresh: %.6f", estimator_lio.min_cov_thresh);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "max_cov_thresh: %.6f", estimator_lio.max_cov_thresh);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "min_cov_bound: %.6f", estimator_lio.min_cov_bound);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "max_cov_bound: %.6f", estimator_lio.max_cov_bound);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "min_plane_cov_thresh: %.6f", estimator_lio.min_plane_cov_thresh);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "max_plane_cov_thresh: %.6f", estimator_lio.max_plane_cov_thresh);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "min_plane_cov_bound: %.6f", estimator_lio.min_plane_cov_bound);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "max_plane_cov_bound: %.6f", estimator_lio.max_plane_cov_bound);
        }
        else {
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "min_cov_thresh: %.6f", estimator_lo.min_cov_thresh);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "max_cov_thresh: %.6f", estimator_lo.max_cov_thresh);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "min_cov_bound: %.6f", estimator_lo.min_cov_bound);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "max_cov_bound: %.6f", estimator_lo.max_cov_bound);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "min_plane_cov_thresh: %.6f", estimator_lo.min_plane_cov_thresh);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "max_plane_cov_thresh: %.6f", estimator_lo.max_plane_cov_thresh);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "min_plane_cov_bound: %.6f", estimator_lo.min_plane_cov_bound);
            RCLCPP_INFO(rclcpp::get_logger("GEORIO"), "max_plane_cov_bound: %.6f", estimator_lo.max_plane_cov_bound);
        }
    }

    void initFilter(int64_t start_t_ns, Eigen::Vector3d t_init = Eigen::Vector3d::Zero(), Eigen::Quaterniond q_init = Eigen::Quaterniond::Identity())
    {
        Eigen::Matrix<double, 24, 24> cov_RCPs = cov_P0 * Eigen::Matrix<double, 24, 24>::Identity();
        Eigen::Matrix<double, 30, 30> Q = Eigen::Matrix<double, 30, 30>::Zero();
        Eigen::Matrix<double, 6, 6> Q_block_old = Eigen::Matrix<double, 6, 6>::Zero();
        Q_block_old.topLeftCorner<3, 3>() = cov_RCP_pos_old*cov_sys_pos *Eigen::Matrix3d::Identity();
        Q_block_old.bottomRightCorner<3, 3>() = cov_RCP_ort_old*cov_sys_ort *Eigen::Matrix3d::Identity();
        Eigen::Matrix<double, 6, 6> Q_block_new = Eigen::Matrix<double, 6, 6>::Zero();
        Q_block_new.topLeftCorner<3, 3>() = cov_RCP_pos_new*cov_sys_pos *Eigen::Matrix3d::Identity();
        Q_block_new.bottomRightCorner<3, 3>() = cov_RCP_ort_new*cov_sys_ort *Eigen::Matrix3d::Identity();        
        Q.topLeftCorner<6, 6>() = Q_block_old;
        Q.block<6, 6>(6, 6) = Q_block_old;
        Q.block<6, 6>(12, 12) = Q_block_old;
        // rows 24..29 = IMU bias block (inherited from RESPLE; intentional)
        Q.bottomRightCorner<6, 6>() = Q_block_new;
        if (if_lidar_only) {
            estimator_lo.setState(dt_ns, start_t_ns, t_init, q_init, Q.topLeftCorner<24, 24>(), cov_RCPs);  
            spline = estimator_lo.getSpline();
        } else {
            Eigen::Matrix<double, 30, 30> cov_x = Eigen::Matrix<double, 30, 30>::Zero();
            cov_x.topLeftCorner<24, 24>() = cov_RCPs;
            cov_x.block<3, 3>(24, 24) = cov_ba.asDiagonal();
            cov_x.block<3, 3>(27, 27) = cov_bg.asDiagonal();            
            estimator_lio.setState(dt_ns, start_t_ns, t_init, q_init, Q, cov_x);   
            spline = estimator_lio.getSpline(); 
        }
    }     

    void getImuCallback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
    {
        m_buff.lock();
        imu_int_buff.push_back(imu_msg);
        m_buff.unlock();        
    }    

    template<typename T>
    void ousterLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr ouster_msg_in, const std::string& name)
    {
        const LidarConfig& lidar = lidars.at(name);        
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());
        typename pcl::PointCloud<T>::Ptr pc_last_ouster(new typename pcl::PointCloud<T>());
        pcl::fromROSMsg(*ouster_msg_in, *pc_last_ouster);
        size_t plsize = pc_last_ouster->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        int64_t time_begin = rclcpp::Time(ouster_msg_in->header.stamp).nanoseconds() - time_offset;
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);
        int64_t max_ofs_ns = 0;
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                pt.x = pc_last_ouster->points[i].x;
                pt.y = pc_last_ouster->points[i].y;
                pt.z = pc_last_ouster->points[i].z;
                pt.intensity = float (pc_last_ouster->points[i].t) / float (1e6); // unit: ms
                pt.curvature = pc_last_ouster->points[i].intensity;
                if (pt.intensity >= 0 && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && pc_last_ouster->points[i].t + time_begin > last_t_ns) {
                    pc_last->points.push_back(pt);
                    int64_t ofs = pc_last_ouster->points[i].t;
                    max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                }
            }
        }
        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();        
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);
    }    

    void livoxLidarCallback(const livox_ros_driver::msg::CustomMsg::SharedPtr livox_msg_in, const std::string& name)
    {
        const LidarConfig& lidar = lidars.at(name);   
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());     
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        int64_t time_begin = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);
        int64_t max_ofs_ns = 0;
        int valid_point_num = 0;
        pcl::PointXYZINormal pt_pre;
        pt_pre.x = livox_msg_in->points[0].x;
        pt_pre.y = livox_msg_in->points[0].y;
        pt_pre.z = livox_msg_in->points[0].z;
        int N_SCAN_LINES = lidar.scan_line;
        float blind = lidar.blind;        
        for (int i = 1; i < plsize; ++i) {
            if ((livox_msg_in->points[i].line < N_SCAN_LINES) && ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00)) {
                valid_point_num++;
                if (valid_point_num % point_filter_num == 0) {
                    pcl::PointXYZINormal pt;
                    pt.x = livox_msg_in->points[i].x;
                    pt.y = livox_msg_in->points[i].y;
                    pt.z = livox_msg_in->points[i].z;
                    pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6); // unit: ms
                    pt.curvature = livox_msg_in->points[i].reflectivity;
                    if (pt.intensity >= 0 && ((abs(pt.x - pt_pre.x) > 1e-7) || (abs(pt.y - pt_pre.y) > 1e-7) || (abs(pt.z - pt_pre.z) > 1e-7))
                                            && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind)&& livox_msg_in->points[i].offset_time + time_begin > last_t_ns) {
                        int64_t ofs = livox_msg_in->points[i].offset_time;
                        max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                        pc_last->points.push_back(pt);
                    }
                    pt_pre = pt;
                }

            } 
        }
        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);
    }    

    void livoxLidar2Callback(const livox_ros_driver2::msg::CustomMsg::SharedPtr livox_msg_in, const std::string& name)
    {
        const LidarConfig& lidar = lidars.at(name);     
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());        
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        int64_t time_begin = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);
        int64_t max_ofs_ns = 0;
        int valid_point_num = 0;
        pcl::PointXYZINormal pt_pre;
        pt_pre.x = livox_msg_in->points[0].x;
        pt_pre.y = livox_msg_in->points[0].y;
        pt_pre.z = livox_msg_in->points[0].z;
        int N_SCAN_LINES = lidar.scan_line;
        float blind = lidar.blind;          
        for (int i = 1; i < plsize; ++i) {
            if ((livox_msg_in->points[i].line < N_SCAN_LINES) && ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00)) {
                valid_point_num++;
                if (valid_point_num % point_filter_num == 0) {
                    pcl::PointXYZINormal pt;
                    pt.x = livox_msg_in->points[i].x;
                    pt.y = livox_msg_in->points[i].y;
                    pt.z = livox_msg_in->points[i].z;
                    pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6); 
                    pt.curvature = livox_msg_in->points[i].reflectivity;
                    if (pt.intensity >= 0 && ((abs(pt.x - pt_pre.x) > 1e-7) || (abs(pt.y - pt_pre.y) > 1e-7) || (abs(pt.z - pt_pre.z) > 1e-7))
                                            && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && livox_msg_in->points[i].offset_time + time_begin > last_t_ns) {
                        int64_t ofs = livox_msg_in->points[i].offset_time;
                        max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                        pc_last->points.push_back(pt);
                    }
                    pt_pre = pt;
                }
            } 
        }
        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();        
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);
    }

    void livoxAVIACallback(const livox_interfaces::msg::CustomMsg::SharedPtr livox_msg_in, const std::string& name)
    {
        const LidarConfig& lidar = lidars.at(name);       
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());      
        int plsize = livox_msg_in->point_num;
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        int64_t time_begin = rclcpp::Time(livox_msg_in->header.stamp).nanoseconds();
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);
        int64_t max_ofs_ns = 0;
        int valid_point_num = 0;
        pcl::PointXYZINormal pt_pre;
        pt_pre.x = livox_msg_in->points[0].x;
        pt_pre.y = livox_msg_in->points[0].y;
        pt_pre.z = livox_msg_in->points[0].z;
        int N_SCAN_LINES = lidar.scan_line;
        float blind = lidar.blind;           
        for (int i = 1; i < plsize; ++i) {
            if ((livox_msg_in->points[i].line < N_SCAN_LINES) && ((livox_msg_in->points[i].tag & 0x30) == 0x10 || (livox_msg_in->points[i].tag & 0x30) == 0x00) && livox_msg_in->points[i].offset_time + time_begin > last_t_ns) {
                valid_point_num++;
                if (valid_point_num % point_filter_num == 0) {
                    pcl::PointXYZINormal pt;
                    pt.x = livox_msg_in->points[i].x;
                    pt.y = livox_msg_in->points[i].y;
                    pt.z = livox_msg_in->points[i].z;
                    pt.intensity = float (livox_msg_in->points[i].offset_time) / float (1e6); 
                    pt.curvature = livox_msg_in->points[i].reflectivity;
                    if (pt.intensity >= 0 && ((abs(pt.x - pt_pre.x) > 1e-7) || (abs(pt.y - pt_pre.y) > 1e-7) ||
                                            (abs(pt.z - pt_pre.z) > 1e-7))
                                            && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind)) {
                        int64_t ofs = livox_msg_in->points[i].offset_time;
                        max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                        pc_last->points.push_back(pt);
                    }
                    pt_pre = pt;
                }
            }
        }
        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);
    }        

    void hesaiLidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr hesai_msg_in, const std::string& name)
	{
        const LidarConfig& lidar = lidars.at(name);    
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());    
        pcl::PointCloud<hesai_ros::Point>::Ptr pc_last_hesai(new pcl::PointCloud<hesai_ros::Point>());
        pcl::fromROSMsg(*hesai_msg_in, *pc_last_hesai);
        size_t plsize = pc_last_hesai->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(hesai_msg_in->header.stamp);
        int64_t time_begin = timestamp_begin.nanoseconds();
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);
        int64_t max_ofs_ns = 0;        
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                pt.x = pc_last_hesai->points[i].x;
                pt.y = pc_last_hesai->points[i].y;
                pt.z = pc_last_hesai->points[i].z;
                double timestamp_s;
                double timestamp_ns = std::modf(pc_last_hesai->points[i].timestamp, &timestamp_s);
                rclcpp::Time timestamp_ros(static_cast<int32_t>(timestamp_s), static_cast<int32_t>(timestamp_ns * 1.0e9),
                    rcl_clock_type_t::RCL_ROS_TIME);
                pt.intensity = (timestamp_ros - timestamp_begin).seconds() * 1.0e3; 
                pt.curvature = pc_last_hesai->points[i].intensity;
                if (pt.intensity >= 0 && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && CommonUtils::ms2ns(pt.intensity) + time_begin > last_t_ns) {
                    int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                    max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                    pc_last->points.push_back(pt);
                }
            }
        }
        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();        
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);
	}     

    void livoxMid360BoxiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr livox_msg_in, const std::string& name)
	{
        const LidarConfig& lidar = lidars.at(name);   
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());     
        pcl::PointCloud<livox_mid360_boxi::Point>::Ptr pc_last_livox(new pcl::PointCloud<livox_mid360_boxi::Point>());
        pcl::fromROSMsg(*livox_msg_in, *pc_last_livox);
        size_t plsize = pc_last_livox->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(livox_msg_in->header.stamp);
        int64_t time_begin = timestamp_begin.nanoseconds();
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);   
        int64_t max_ofs_ns = 0;         
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                pt.x = pc_last_livox->points[i].x;
                pt.y = pc_last_livox->points[i].y;
                pt.z = pc_last_livox->points[i].z;
                rclcpp::Time timestamp_ros(static_cast<int64_t>(pc_last_livox->points[i].timestamp),
                    rcl_clock_type_t::RCL_ROS_TIME);
                pt.intensity = (timestamp_ros - timestamp_begin).seconds() * 1.0e3;
                pt.curvature = pc_last_livox->points[i].intensity;
                if (pt.intensity >= 0 && pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind) && CommonUtils::ms2ns(pt.intensity) + time_begin > last_t_ns) {
                    int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                    max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                    pc_last->points.push_back(pt);
                }
            }
        }
        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();     
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);   
	} 

    void snailContiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr conti_msg_in, const std::string& name)
	{
        // auto T1 = std::chrono::steady_clock::now();

        const LidarConfig& lidar = lidars.at(name);   
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());     
        pcl::PointCloud<snail_conti::Point>::Ptr pc_last_conti(new pcl::PointCloud<snail_conti::Point>());
        pcl::fromROSMsg(*conti_msg_in, *pc_last_conti);
        size_t plsize = pc_last_conti->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(conti_msg_in->header.stamp);
        int64_t time_begin = timestamp_begin.nanoseconds();
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);   
        int64_t max_ofs_ns = 0;         
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        float max_range = lidar.max_range;
        float min_z = lidar.min_z;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                pt.x = pc_last_conti->points[i].x;
                pt.y = pc_last_conti->points[i].y;
                pt.z = pc_last_conti->points[i].z;
                pt.intensity = 0;
                // TODO : raw intensity is in [0, 255] unit: dB
                // TODO : need type casting?
                pt.curvature = pc_last_conti->points[i].intensity; 
                pt.normal_z = pc_last_conti->points[i].doppler;
                 

                if ((pt.intensity >= 0) && (pt.z > min_z) && (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind)) && (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z < (max_range * max_range)) ) {
                    // int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                    // max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                    pc_last->points.push_back(pt);
                }
            }
        }

        const auto& ransac_config = radar_ransac_configs.at(name);

        const auto radar_result = filterRadarStaticPoints(pc_last, ransac_config);
        if (!radar_result.success) {
            return;
        }

        // Eigen::Vector3d ego_velocity = -radar_result.ego_vel;
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Timestamp: %lf", timestamp_begin.seconds());
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Ego velocity: %f, %f, %f", ego_velocity.x(), ego_velocity.y(), ego_velocity.z());
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Raw points size: %d", static_cast<int>(radar_result.raw_point_count));
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Inlier points size: %d", static_cast<int>(radar_result.static_point_count));

        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();     
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);

        // auto T2 = std::chrono::steady_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(T2 - T1).count();
        // std::ofstream timepre_file("time_pre.txt", std::ios::app);
        // timepre_file.setf(std::ios::fixed, std::ios::floatfield);
        // timepre_file.precision(9);
        // timepre_file << time_begin << "," << duration << std::endl;
        // timepre_file.close();
	}

    void huginCallback(const sensor_msgs::msg::PointCloud2::SharedPtr hugin_msg_in, const std::string& name)
	{
        const LidarConfig& lidar = lidars.at(name);
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());
        pcl::PointCloud<hugin::Point>::Ptr pc_last_hugin(new pcl::PointCloud<hugin::Point>());
        pcl::fromROSMsg(*hugin_msg_in, *pc_last_hugin);
        const size_t plsize = pc_last_hugin->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        const int64_t time_begin = rclcpp::Time(hugin_msg_in->header.stamp).nanoseconds();
        int64_t max_ofs_ns = 0;
        pcl::PointXYZINormal pt;
        const float blind = lidar.blind;
        const float max_range = lidar.max_range;
        const float min_z = lidar.min_z;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                const auto& hugin_pt = pc_last_hugin->points[i];
                pt.x = hugin_pt.x;
                pt.y = hugin_pt.y;
                pt.z = hugin_pt.z;
                pt.intensity = 0;
                pt.curvature = hugin_pt.power;
                pt.normal_z = hugin_pt.doppler;

                if ((pt.intensity >= 0) && (pt.z > min_z) &&
                    (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z > (blind * blind)) &&
                    (pt.x*pt.x + pt.y*pt.y + pt.z*pt.z < (max_range * max_range))) {
                    pc_last->points.push_back(pt);
                }
            }
        }

        const auto& ransac_config = radar_ransac_configs.at(name);
        const auto radar_result = filterRadarStaticPoints(pc_last, ransac_config);
        if (!radar_result.success) {
            return;
        }

        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);
	}

    void herculesContiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr conti_msg_in, const std::string& name)
	{
        // auto T1 = std::chrono::steady_clock::now();
        
        const LidarConfig& lidar = lidars.at(name);   
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr pc_last(new pcl::PointCloud<pcl::PointXYZINormal>());     
        pcl::PointCloud<hercules_conti::Point>::Ptr pc_last_conti(new pcl::PointCloud<hercules_conti::Point>());
        pcl::fromROSMsg(*conti_msg_in, *pc_last_conti);
        size_t plsize = pc_last_conti->size();
        if (plsize == 0) return;
        pc_last->reserve(plsize);
        rclcpp::Time timestamp_begin = rclcpp::Time(conti_msg_in->header.stamp);
        int64_t time_begin = timestamp_begin.nanoseconds();
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);   
        int64_t max_ofs_ns = 0;         
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        float max_range = lidar.max_range;
        float min_z = lidar.min_z;
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                pt.x = pc_last_conti->points[i].x;
                pt.y = pc_last_conti->points[i].y;
                pt.z = pc_last_conti->points[i].z;
                pt.intensity = 0;
                // raw intensity is in [-128, 127], which is RCS value in dB
                pt.curvature = double(pc_last_conti->points[i].RCS) + 128.0; // make positive
                pt.normal_z = pc_last_conti->points[i].v;

                if ((pt.intensity >= 0) && (pt.z > min_z) && (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z > (blind * blind)) && (pt.x*pt.x+pt.y*pt.y+pt.z*pt.z < (max_range * max_range)) ) {
                    // int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                    // max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                    pc_last->points.push_back(pt);
                }
            }
        }

        

        const auto& ransac_config = radar_ransac_configs.at(name);

        const auto radar_result = filterRadarStaticPoints(pc_last, ransac_config);
        if (!radar_result.success) {
            return;
        }

        // Eigen::Vector3d ego_velocity = -radar_result.ego_vel;
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Timestamp: %lf", timestamp_begin.seconds());
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Ego velocity: %f, %f, %f", ego_velocity.x(), ego_velocity.y(), ego_velocity.z());
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Raw points size: %d", static_cast<int>(radar_result.raw_point_count));
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Inlier points size: %d", static_cast<int>(radar_result.static_point_count));

        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();     
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);

        // auto T2 = std::chrono::steady_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(T2 - T1).count();
        // std::ofstream timepre_file("time_pre.txt", std::ios::app);
        // timepre_file.setf(std::ios::fixed, std::ios::floatfield);
        // timepre_file.precision(9);
        // timepre_file << time_begin << "," << duration << std::endl;
        // timepre_file.close();
	}

    void hkustContiCallback(const sensor_msgs::msg::PointCloud2::SharedPtr conti_msg_in, const std::string& name)
	{
        const LidarConfig& lidar = lidars.at(name);   
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
        int64_t last_t_ns = getLastLidarTimestampNs(name, time_begin);   
        int64_t max_ofs_ns = 0;         
        pcl::PointXYZINormal pt;
        float blind = lidar.blind;
        float max_range = lidar.max_range;
        float min_z = lidar.min_z;
        const uint8_t *data_ptr = conti_msg_in->data.data();
        for (unsigned int i = 0; i < plsize; ++i) {
            if (i % point_filter_num == 0) {
                const uint8_t *pt_base = data_ptr + i * pointBytes;
                double azimuth     = *reinterpret_cast<const float*>(pt_base + offsetAzimuth);
                double elevation   = *reinterpret_cast<const float*>(pt_base + offsetElevation);
                double range       = *reinterpret_cast<const float*>(pt_base + offsetRange);
                pt.x = range * cos(azimuth) * cos(elevation);
                pt.y = range * sin(azimuth) * cos(elevation);
                pt.z = range * sin(elevation);
                pt.intensity = 0;
                // RCLCPP_INFO(rclcpp::get_logger("radar"), "std values: azimuth: %f, elevation: %f, range: %f, velocity: %f", 
                //             *reinterpret_cast<const float*>(pt_base + offsetAzimuthSTD),
                //             *reinterpret_cast<const float*>(pt_base + offsetElevationSTD),
                //             *reinterpret_cast<const float*>(pt_base + offsetRangeSTD),
                //             *reinterpret_cast<const float*>(pt_base + offsetVelocitySTD));
                // TODO : raw intensity is in [0, 255] unit: dB
                // TODO : need type casting?
                // RCLCPP_INFO(rclcpp::get_logger("radar"), "HKUST Continental point rcs: %d", *reinterpret_cast<const int8_t*>(pt_base + offsetRCS));
                pt.curvature = double(*reinterpret_cast<const int8_t*>(pt_base + offsetRCS)) + 128.0; // make positive
                pt.normal_z = *reinterpret_cast<const float*>(pt_base + offsetVelocity);
                 

                if ((pt.intensity >= 0) && (pt.z > min_z) && (range * range > (blind * blind)) && (range * range < (max_range * max_range)) ) {
                    // int64_t ofs = CommonUtils::ms2ns(pt.intensity);
                    // max_ofs_ns = max_ofs_ns > ofs ? max_ofs_ns : ofs;
                    pc_last->points.push_back(pt);
                }
            }
        }

        const auto& ransac_config = radar_ransac_configs.at(name);

        const auto radar_result = filterRadarStaticPoints(pc_last, ransac_config);
        if (!radar_result.success) {
            return;
        }

        // Eigen::Vector3d ego_velocity = -radar_result.ego_vel;
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Timestamp: %lf", timestamp_begin.seconds());
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Ego velocity: %f, %f, %f", ego_velocity.x(), ego_velocity.y(), ego_velocity.z());
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Raw points size: %d", static_cast<int>(radar_result.raw_point_count));
        // RCLCPP_INFO(rclcpp::get_logger("radar"), "Inlier points size: %d", static_cast<int>(radar_result.static_point_count));

        LidarData& lidar_buffs = lidars_data.at(name);
        lidar_buffs.mtx_pc.lock();
        lidar_buffs.pc_buff.push_back(pc_last->points);
        lidar_buffs.t_buff.push_back(time_begin);
        lidar_buffs.mtx_pc.unlock();     
        setLastLidarTimestampNs(name, time_begin + max_ofs_ns);
	}

    // publish current scan in world frame
    void publishFrameWorld() 
    {
        int size = pc_world.points.size();
        pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudWorld(new pcl::PointCloud<pcl::PointXYZI>(size, 1));
        for (int i = 0; i < size; i++) {
            laserCloudWorld->points[i].x = pc_world.points[i].x;
            laserCloudWorld->points[i].y = pc_world.points[i].y;
            laserCloudWorld->points[i].z = pc_world.points[i].z;
            laserCloudWorld->points[i].intensity = pc_world.points[i].curvature; 
        }
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = rclcpp::Time(spline->maxTimeNs());
        laserCloudmsg.header.frame_id = odom_id;
        pub_cur_scan->publish(laserCloudmsg);
    }

    bool initialization()
    {
        if (if_init_filter && if_init_map) {
            return true;
        } 
        for (const auto& [lidar_name, lidar_data] : lidars_data) {
            if (lidar_data.pt_buff.empty()) {
                return false;
            }
        }
        if (imu_buff.empty() && !if_lidar_only) {
            return false;
        }
        RCLCPP_INFO(rclcpp::get_logger("initializer"), "Initialization started");
        int64_t start_t_ns = std::numeric_limits<int64_t>::max();
        for (const auto& [lidar_name, lidar_data] : lidars_data) {
            start_t_ns = std::min(start_t_ns, std::max(lidar_data.pt_buff.front().time_ns, int64_t(0)));
        }        
        if (!if_init_filter) {
              
            if (!if_lidar_only) {
                Eigen::Vector3d gravity_sum(0, 0, 0);
                m_buff.lock();
                int n_imu = std::min(10, int(imu_buff.size()));
                for (int i = 0; i < n_imu; i++) {
                    gravity_sum += imu_buff.at(i).accel;
                }
                while (!imu_buff.empty() && imu_buff.front().time_ns < start_t_ns) {
                    imu_buff.pop_front();
                }                    
                m_buff.unlock();
                gravity_sum /= n_imu;
                Eigen::Vector3d gravity_ave = gravity_sum.normalized() * 9.81;
                Eigen::Matrix3d R0 = CommonUtils::g2R(gravity_ave);
                double yaw = CommonUtils::R2ypr(R0).x();
                double pitch = CommonUtils::R2ypr(R0).y();
                double roll = CommonUtils::R2ypr(R0).z();
                RCLCPP_INFO(rclcpp::get_logger("initializer"), "Yaw: %.3f, Pitch: %.3f, Roll: %.3f", yaw, pitch, roll);
                R0 = CommonUtils::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
                Eigen::Quaterniond q0(R0);
                q_WI = Quater::positify(q0);
                gravity = q_WI * gravity_ave;
                // gravity = gravity_ave;
            }
            RCLCPP_INFO(rclcpp::get_logger("initializer"), "Gravity: %.3f, %.3f, %.3f", gravity.x(), gravity.y(), gravity.z());
            initFilter(start_t_ns, Eigen::Vector3d(0, 0, 0), q_WI);
            // initFilter(start_t_ns, Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
            if_init_filter = true;            
            std_msgs::msg::Int64 start_time;
            start_time.data = start_t_ns;
            pub_start_time->publish(start_time);    
        }
        if (!if_init_map) {
            if (if_lidar_only) {
                estimator_lo.propRCP(start_t_ns);  
            } else {
                estimator_lio.propRCP(start_t_ns);
            }
            int feats_down_size = 0;
            for (const auto& [lidar_name, lidar_data] : lidars_data) {
                for (size_t i = 0; i < lidar_data.pt_buff.size(); i++) {
                    if (lidar_data.pt_buff[i].time_ns < start_t_ns + 1e8) {
                        feats_down_size++;
                    } else {
                        break;
                    }
                }
            }

            if(feats_down_size < 100) {
                RCLCPP_WARN(rclcpp::get_logger("initializer"), "Not enough points for initialization: %d", feats_down_size);
                return false;
            }

            // hkust
            // if(feats_down_size < 50) {
            //     RCLCPP_WARN(rclcpp::get_logger("initializer"), "Not enough points for initialization: %d", feats_down_size);
            //     return false;
            // }
            
            // ikdtree version
            if(ikdtree.Root_Node == nullptr) {
                ikdtree.set_downsample_param(ds_lm_voxel);
            }

            // for map comparison with GT pose
            if(ikdtree_gt.Root_Node == nullptr) {
                ikdtree_gt.set_downsample_param(ds_lm_voxel);
            }





            pc_world.clear();
            pc_world.resize(feats_down_size);

            pt_data_buff.clear();
            // pt_data_buff.resize(feats_down_size);

            RCLCPP_INFO(rclcpp::get_logger("initializer"), "start time: %.9f",
                            static_cast<double>(start_t_ns + 1e8) * 1e-9);
            int world_i = 0;
            for (auto& [lidar_name, lidar_data] : lidars_data) {
                for (size_t i = 0; i < lidar_data.pt_buff.size(); i++) {
                    if (lidar_data.pt_buff[i].time_ns < start_t_ns + 1e8) {

                        PointData& pt_data = lidar_data.pt_buff[i];

                        voxelmap_manager->calcBodyCov(pt_data.pt_l, pt_data.cov);
                        // kd tree version
                        Association::pointLocalToWorld(start_t_ns, spline, pt_data, pt_data);
                        pc_world.points[world_i] = pt_data.pt_w;
                            
                        // Association::pointBodyToWorld(start_t_ns, spline, lidar_data.pt_buff[i].pt,
                        //     pc_world.points[world_i], lidar_data.pt_buff[i].t_bl, lidar_data.pt_buff[i].q_bl);

                        // voxelmap version
                        // pt_data_buff.emplace_back(lidar_data.pt_buff[i]);
                        // Association::pointLocalToWorld(start_t_ns, spline, pt_data_buff[world_i], pt_data_buff[world_i]);

                        world_i++;
                    } else {
                        break;
                    }
                }

            }

            ikdtree.Build(pc_world.points);
            // ikdtree_gt.Build(pc_world.points);
            RCLCPP_INFO(rclcpp::get_logger("initializer"), "ikdtree built");

            // voxelmap_manager->pt_data_buff_.swap(pt_data_buff);
            // voxelmap_manager->buildVoxelMap(start_t_ns, spline);
            // RCLCPP_INFO(rclcpp::get_logger("initializer"), "voxelmap built");

            pc_world.clear();
            for (auto& [lidar_name, lidar_data] : lidars_data) {
                while (!lidar_data.pt_buff.empty() && lidar_data.pt_buff.front().time_ns < start_t_ns + 1e8) {
                    lidar_data.pt_buff.pop_front();
                }
            }
            RCLCPP_INFO(rclcpp::get_logger("initializer"), "Number of knots in spline: %zu", spline->numKnots());
            RCLCPP_INFO(rclcpp::get_logger("initializer"), "Number of points init: %d", feats_down_size);
            RCLCPP_INFO(rclcpp::get_logger("initializer"), "Initialization ended");
            if_init_map = true;
        }       
        return false; 
    }

    bool collectMeasurements()
    {
        int64_t pt_min_time = std::numeric_limits<int64_t>::max();
        int64_t pt_max_time = std::numeric_limits<int64_t>::max();            
        for (const auto& [lidar_name, lidar_data] : lidars_data) {
            if (lidar_data.pt_buff.empty()) {
                return false;
            }
            pt_min_time = std::min(pt_min_time, lidar_data.pt_buff.front().time_ns);
            pt_max_time = std::min(pt_max_time, lidar_data.pt_buff.back().time_ns);
        }
        if (pt_max_time <= spline->maxTimeNs() + dt_ns) { // more points need for new propagation
            return false;
        }      
        if (!if_lidar_only && (imu_buff.empty() || imu_buff.back().time_ns <= spline->maxTimeNs())) { // more imu need for new propagation
            return false;
        }
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "***************************");
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "* Collecting measurements *");
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "***************************");
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "pt_buff min time : %.9f", double(pt_min_time) * 1e-9);
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "pt_buff max time : %.9f", double(pt_max_time) * 1e-9);
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "Spline max time  : %.9f", double(spline->maxTimeNs()) * 1e-9);

        
        int64_t max_time_ns = std::min(spline->maxTimeNs(), pt_min_time + dt_ns);
        if (pt_min_time > max_time_ns) { // if maxtime < pt_min_time < pt_min_time + dt_ns
            if (if_lidar_only) {
                estimator_lo.propRCP(pt_min_time);
            } else {
                estimator_lio.propRCP(pt_min_time); // while maxtime < pt_min_time
            }                
            max_time_ns = spline->maxTimeNs();
        }

        // int64_t max_time_ns = spline->maxTimeNs();
        // if (pt_min_time > max_time_ns) { // if pt_min_time < maxtime < pt_min_time + dt_ns
        //     if (if_lidar_only) {
        //         estimator_lo.propRCP(pt_min_time);
        //     } else {
        //         estimator_lio.propRCP(pt_min_time); // while maxtime < pt_min_time
        //     }                
        //     max_time_ns = spline->maxTimeNs();
        // }

        if (spline->numKnots() > 4) {
            max_time_ns = spline->maxTimeNs();
        }

        int cnt = 0;
        for (auto& [lidar_name, lidar_data] : lidars_data) {
            while (!lidar_data.pt_buff.empty() && lidar_data.pt_buff.front().time_ns <= max_time_ns &&
                    cnt < num_points_upd) {
                if (spline->numKnots() < 10 || lidar_data.pt_buff.front().time_ns >= spline->maxTimeNs() - dt_ns) {
                    // auto fmt = Eigen::IOFormat(3, 0, ", ", "\n", "[", "]");
                    // RCLCPP_INFO(rclcpp::get_logger("collect"), "cov inited\n%s",
                    //         CommonUtils::eigenToString(lidar_data.pt_buff.front().cov, fmt).c_str());
                    // RCLCPP_INFO(rclcpp::get_logger("collect"), "pt_l: %f, %f, %f",
                    //         lidar_data.pt_buff.front().pt_l.x(), lidar_data.pt_buff.front().pt_l.y(),
                    //         lidar_data.pt_buff.front().pt_l.z());


                    // radar uncertainty
                    voxelmap_manager->calcBodyCov(lidar_data.pt_buff.front().pt_l, lidar_data.pt_buff.front().cov);



                    // RCLCPP_INFO(rclcpp::get_logger("collect"), "cov calculated\n%s",
                    //         CommonUtils::eigenToString(lidar_data.pt_buff.front().cov, fmt).c_str());

                    // Eigen::Matrix3d JRangeThetaPhi;
                    // double Range = lidar_data.pt_buff.front().pt_l.norm();
                    // double Theta = acos(lidar_data.pt_buff.front().pt_l.z() / Range);
                    // double Phi = atan2(lidar_data.pt_buff.front().pt_l.y(), lidar_data.pt_buff.front().pt_l.x());

                    // RCLCPP_INFO(rclcpp::get_logger("collect"), "Range: %f, Theta: %f, Phi: %f",
                    //         Range, Theta, Phi);

                    // JRangeThetaPhi << cos(Theta) * cos(Phi), -Range * sin(Theta) * cos(Phi),
                    // -Range * cos(Theta) * sin(Phi), sin(Theta) * cos(Phi),
                    // Range * cos(Theta) * cos(Phi), -Range * sin(Theta) * sin(Phi), sin(Phi),
                    // 0, -Range * cos(Phi);

                    // RCLCPP_INFO(rclcpp::get_logger("collect"), "J\n%s",
                    //         CommonUtils::eigenToString(JRangeThetaPhi, fmt).c_str());
                    
                    // Eigen::Matrix3d Sigma;
                    // Sigma << voxelmap_manager->config_setting_.range_acc_ * voxelmap_manager->config_setting_.range_acc_, 0, 0,
                    // 0, DEG2RAD(voxelmap_manager->config_setting_.azi_acc_) * DEG2RAD(voxelmap_manager->config_setting_.azi_acc_), 0,
                    // 0, 0, DEG2RAD(voxelmap_manager->config_setting_.elev_acc_) * DEG2RAD(voxelmap_manager->config_setting_.elev_acc_);

                    // RCLCPP_INFO(rclcpp::get_logger("collect"), "Sigma\n%s",
                    //         CommonUtils::eigenToString(Sigma, fmt).c_str());

                    // Eigen::Matrix3d tmp_cov = JRangeThetaPhi * Sigma * JRangeThetaPhi.transpose();

                    // RCLCPP_INFO(rclcpp::get_logger("collect"), "cov calculated new\n%s",
                    //         CommonUtils::eigenToString(tmp_cov, fmt).c_str());

                    pt_meas.emplace_back(lidar_data.pt_buff.front());
                }
                lidar_data.pt_buff.pop_front();
                cnt++;
            }
        }                    
        if (!if_lidar_only) {
            while (!imu_buff.empty() && imu_buff.front().time_ns < spline->minTimeNs()) {
                imu_buff.pop_front();
            }                
            while (!imu_buff.empty() && imu_buff.front().time_ns <= max_time_ns) {
                imu_meas.emplace_back(imu_buff.front());
                imu_buff.pop_front();
            }
            if(imu_meas.empty()) {
            //     pt_meas.clear();
                return false;
            }
        }

        // RCLCPP_INFO(rclcpp::get_logger("collect"), "Collected %zu lidar points and %zu imu measurements",
        //     pt_meas.size(), imu_meas.size());

        // RCLCPP_INFO(rclcpp::get_logger("collect"), "Spline max time: %.9f", double(spline->maxTimeNs()) * 1e-9);
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "pt min time    : %.9f", double(pt_meas.front().time_ns) * 1e-9);
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "pt max time    : %.9f", double(pt_meas.back().time_ns) * 1e-9);
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "imu min time   : %.9f", double(imu_meas.front().time_ns) * 1e-9);
        // RCLCPP_INFO(rclcpp::get_logger("collect"), "imu max time   : %.9f", double(imu_meas.back().time_ns) * 1e-9);

        // ImuData imu_data_ = imu_meas.back();
        // if (imu_data_.gyro(0) > 0.4 || imu_data_.gyro(1) > 0.4 || imu_data_.gyro(2) > 0.4) {
        //     RCLCPP_WARN(rclcpp::get_logger("collect"), "High gyro values detected: %.3f, %.3f, %.3f",
        //         imu_data_.gyro(0), imu_data_.gyro(1), imu_data_.gyro(2));
        
        //     if (imu_data_.gyro(0) > 0.4){
        //         param.cov_gyro(0) = 0.01;
        //     }
        //     if (imu_data_.gyro(1) > 0.4){
        //         param.cov_gyro(1) = 0.01;
        //     }
        //     if (imu_data_.gyro(2) > 0.4){
        //         param.cov_gyro(2) = 0.01;
        //     }
        // }
        // else {
        //     param.cov_gyro = Eigen::Vector3d(0.1, 0.1, 0.1);
        // }
        return true;
 
    }

    Eigen::Vector3d getPositionLiDAR(int64_t t_ns, const Eigen::Vector3d& t_bl)
    {
        if (if_lidar_only) {
            estimator_lo.propRCP(t_ns);
        } else {
            estimator_lio.propRCP(t_ns);
        }
        Eigen::Quaterniond orient_interp;
        Eigen::Vector3d t_interp = spline->itpPosition(t_ns);
        spline->itpQuaternion(t_ns, &orient_interp);
        Eigen::Vector3d t = orient_interp * t_bl + t_interp;
        return t;
    }       

    // remove old points in local map
    void lasermapFovSegment()
    {
        static bool Localmap_Initialized = false;
        cub_needrm.shrink_to_fit();
        Eigen::Vector3d pos_lidar_min(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max());
        Eigen::Vector3d pos_lidar_max(std::numeric_limits<double>::min(), std::numeric_limits<double>::min(),
                std::numeric_limits<double>::min());
        for (const auto& [lidar_name, lidar] : lidars) {
            Eigen::Vector3d pos_lidar = getPositionLiDAR(spline->maxTimeNs(), lidar.t_bl);
            pos_lidar_min = pos_lidar_min.array().min(pos_lidar.array()).matrix();
            pos_lidar_max = pos_lidar_max.array().max(pos_lidar.array()).matrix();
        }        
        if (!Localmap_Initialized){
            for (int i = 0; i < 3; i++){
                LocalMap_Points.vertex_min[i] = pos_lidar_min(i) - cube_len / 2.0;
                LocalMap_Points.vertex_max[i] = pos_lidar_max(i) + cube_len / 2.0;                
            }
            Localmap_Initialized = true;
            return;
        }
        float dist_to_map_edge[3][2];
        bool need_move = false;
        for (int i = 0; i < 3; i++){
            dist_to_map_edge[i][0] = fabs(pos_lidar_min(i) - LocalMap_Points.vertex_min[i]);
            dist_to_map_edge[i][1] = fabs(pos_lidar_max(i) - LocalMap_Points.vertex_max[i]);            
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * det_range || dist_to_map_edge[i][1] <= MOV_THRESHOLD * det_range) need_move = true;
        }
        if (!need_move) return;
        BoxPointType New_LocalMap_Points, tmp_boxpoints;
        New_LocalMap_Points = LocalMap_Points;
        float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * det_range) * 0.5 * 0.9, double(det_range * (MOV_THRESHOLD -1)));
        for (int i = 0; i < 3; i++){
            tmp_boxpoints = LocalMap_Points;
            if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * det_range){
                New_LocalMap_Points.vertex_max[i] -= mov_dist;
                New_LocalMap_Points.vertex_min[i] -= mov_dist;
                tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
                cub_needrm.emplace_back(tmp_boxpoints);
            } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * det_range){
                New_LocalMap_Points.vertex_max[i] += mov_dist;
                New_LocalMap_Points.vertex_min[i] += mov_dist;
                tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
                cub_needrm.emplace_back(tmp_boxpoints);
            }
        }
        LocalMap_Points = New_LocalMap_Points;

        if(cub_needrm.size() > 0) {
            ikdtree.Delete_Point_Boxes(cub_needrm);
        }
    }

    // increment kdtree map
    void mapIncremental()
    {
        Eigen::aligned_vector<pcl::PointXYZINormal> PointToAdd;
        Eigen::aligned_vector<pcl::PointXYZINormal> PointNoNeedDownsample;
        // Eigen::aligned_vector<pcl::PointXYZINormal> PointToAddGT; // for map comparison with GT pose
        // Eigen::aligned_vector<pcl::PointXYZINormal> PointNoNeedDownsampleGT; // for map comparison with GT pose
        int feats_down_size = pc_world.points.size();
        PointToAdd.reserve(feats_down_size);
        PointNoNeedDownsample.reserve(feats_down_size);
        // PointToAddGT.reserve(feats_down_size);
        // PointNoNeedDownsampleGT.reserve(feats_down_size); // for map comparison with GT pose
        for(int i = 0; i < feats_down_size; i++) {     
            const pcl::PointXYZINormal& point = pc_world.points[i];

            if (point.normal_x > 1.5) {
                continue;
            }

            if (!accum_nearest_points[i].empty()) {
                const Eigen::aligned_vector<pcl::PointXYZINormal> &points_near = accum_nearest_points[i];
                bool need_add = true;
                pcl::PointXYZINormal downsample_result, mid_point; 
                
                mid_point.x = floor(point.x/ds_lm_voxel)*ds_lm_voxel + 0.5 * ds_lm_voxel;
                mid_point.y = floor(point.y/ds_lm_voxel)*ds_lm_voxel + 0.5 * ds_lm_voxel;
                mid_point.z = floor(point.z/ds_lm_voxel)*ds_lm_voxel + 0.5 * ds_lm_voxel;
                if (fabs(points_near[0].x - mid_point.x) > 0.5 * ds_lm_voxel || fabs(points_near[0].y - mid_point.y) > 0.5 * ds_lm_voxel || fabs(points_near[0].z - mid_point.z) > 0.5 * ds_lm_voxel){
                    PointNoNeedDownsample.emplace_back(pc_world.points[i]);
                    // PointNoNeedDownsampleGT.emplace_back(pc_world_gtpose.points[i]);
                    continue;
                }
                for (size_t readd_i = 0; readd_i < points_near.size(); readd_i ++) {
                    if (points_near.size() < 5)
                        break;
                    if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * ds_lm_voxel && fabs(points_near[readd_i].y - mid_point.y) < 0.5 * ds_lm_voxel && fabs(points_near[readd_i].z - mid_point.z) < 0.5 * ds_lm_voxel) {
                        need_add = false;
                        break;
                    }
                }
                if (need_add){
                    PointToAdd.emplace_back(point);

                    // for map comparison with GT pose
                    // PointToAddGT.emplace_back(pc_world_gtpose.points[i]);
                }
            } else {
                PointNoNeedDownsample.emplace_back(point);
                // PointNoNeedDownsampleGT.emplace_back(pc_world_gtpose.points[i]); // for map comparison with GT pose

            }
        }
        ikdtree.Add_Points(PointToAdd, true);
        ikdtree.Add_Points(PointNoNeedDownsample, false);

        // for map comparison with GT pose
        // ikdtree_gt.Add_Points(PointToAddGT, true);
        // ikdtree_gt.Add_Points(PointNoNeedDownsampleGT, false);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto nh = rclcpp::Node::make_shared("GEORIO");
    const std::string offline_bag = CommonUtils::readParam<std::string>(nh, "offline_bag", "");
    const std::string offline_hercules_dir = CommonUtils::readParam<std::string>(nh, "offline_hercules_dir", "");
    if (!offline_bag.empty() || !offline_hercules_dir.empty()) {
        {
            GEORIO georio(nh, true);
            if (!offline_bag.empty()) georio.runOffline(offline_bag);
            else georio.runOfflineHercules(offline_hercules_dir);
        }
        voxelmap_manager.reset();  // global; holds a publisher, so it must go before shutdown
        nh.reset();
        rclcpp::shutdown();
        return 0;
    }
    GEORIO georio(nh);
    RCLCPP_INFO_STREAM(nh->get_logger(), "GEORIO starts!");
    rclcpp::Rate rate(2000);
    std::thread opt{&GEORIO::processData, &georio};
    while (rclcpp::ok()) {
        rclcpp::spin_some(nh);
        rate.sleep();
    }
    opt.join();
    voxelmap_manager.reset();
    rclcpp::shutdown();
}
