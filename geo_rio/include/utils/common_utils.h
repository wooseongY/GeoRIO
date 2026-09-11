#pragma once

#include <rclcpp/rclcpp.hpp>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <filesystem>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <omp.h>

#include "utils/eigen_utils.hpp"

inline int NUM_OF_THREAD = 5;
inline int NUM_MATCH_POINTS = 5;


struct ImuData {
    int64_t time_ns;
    Eigen::Vector3d gyro;
    Eigen::Vector3d accel;
    Eigen::Matrix<double, 6, 24> H;
    Eigen::Matrix<double, 6, 1> imu_itp;
    double z_grav = 0.0;
    Eigen::Matrix<double, 1, 30> H_grav = Eigen::Matrix<double, 1, 30>::Zero();
    double omega_unc = 0.0; // uncertainty of the rotation
    double acc_unc = 0.0; // uncertainty of the acceleration

    ImuData(){}
    ImuData(const int64_t s, const Eigen::Vector3d& w, const Eigen::Vector3d& a)
      : time_ns(s), gyro(w), accel(a) {}
    ImuData(const ImuData& other) : time_ns(other.time_ns), gyro(other.gyro), accel(other.accel),
        H(other.H), imu_itp(other.imu_itp), z_grav(other.z_grav), H_grav(other.H_grav),
        omega_unc(other.omega_unc), acc_unc(other.acc_unc) {

        }          
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PoseData {
    int64_t time_ns;
    Eigen::Quaterniond orient;
    Eigen::Vector3d pos;
    PoseData(){}
    PoseData(int64_t s, Eigen::Quaterniond& q, Eigen::Vector3d& t) : time_ns(s), orient(q), pos(t) {}
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

namespace ouster_ros {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    }EIGEN_ALIGN16;
}  

POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (std::uint32_t, t, t)
)

namespace hesai_ros {
  struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}
POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint16_t, ring, ring)
    (double, timestamp, timestamp)
)

namespace livox_mid360_boxi {
    struct EIGEN_ALIGN16 Point {
      PCL_ADD_POINT4D;
      float intensity;
      uint8_t tag;
      uint8_t line;
      double timestamp;
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}
POINT_CLOUD_REGISTER_POINT_STRUCT(livox_mid360_boxi::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (uint8_t, tag, tag)
    (uint8_t, line, line)
    (double, timestamp, timestamp)
)

namespace snail_conti {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        float intensity;
        float doppler;
        float range_std;
        float azimuth_std;
        float elevation_std;
        float doppler_std;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace snail_conti
POINT_CLOUD_REGISTER_POINT_STRUCT(snail_conti::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, doppler, doppler)
    (float, range_std, range_std)
    (float, azimuth_std, azimuth_std)
    (float, elevation_std, elevation_std)
    (float, doppler_std, doppler_std)
)

namespace hugin {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        float range;
        float elevation;
        float azimuth;
        float power;
        float doppler;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace hugin
POINT_CLOUD_REGISTER_POINT_STRUCT(hugin::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, range, range)
    (float, elevation, elevation)
    (float, azimuth, azimuth)
    (float, power, power)
    (float, doppler, doppler)
)

namespace hercules_conti {
    struct EIGEN_ALIGN16 Point {
        PCL_ADD_POINT4D;
        PCL_ADD_INTENSITY
    float v; 
    float r;
    std::int8_t RCS; 
    float azimuth;
    float elevation;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace hercules_conti
POINT_CLOUD_REGISTER_POINT_STRUCT(hercules_conti::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, v, v)
    (float, r, r)
    (std::int8_t, RCS, RCS)
    (float, azimuth, azimuth)
    (float, elevation, elevation) 
)

namespace hkust_conti {
    struct EIGEN_ALIGN16 Point {
        float azimuth;
        float azimuthSTD;
        float elevation;
        float elevationSTD;
        float range;
        float rangeSTD;
        float velocity;
        float velocitySTD;
        std::int8_t rcs;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace hkust_conti
POINT_CLOUD_REGISTER_POINT_STRUCT(hkust_conti::Point,
    (float, azimuth, azimuth)
    (float, azimuthSTD, azimuthSTD)
    (float, elevation, elevation)
    (float, elevationSTD, elevationSTD)
    (float, range, range)
    (float, rangeSTD, rangeSTD)
    (float, velocity, velocity)
    (float, velocitySTD, velocitySTD)
    (std::int8_t, rcs, rcs)
)

class CommonUtils
{
public:

    template <typename T>
    static T readParam(rclcpp::Node::SharedPtr &n, std::string name)
    {
        T ans;
        if (!n->has_parameter(name)) {
            n->declare_parameter<T>(name);
        }        
        if (!n->get_parameter(name, ans)) {
            RCLCPP_FATAL_STREAM(n->get_logger(), "Failed to load " << name);
            exit(1);
        }
        return ans;
    }

    template <typename T>
    static T readParam(rclcpp::Node::SharedPtr &n, std::string name, const T& alternative)
    {
        T ans;
        if (!n->has_parameter(name)) {
            n->declare_parameter<T>(name, alternative);
        }
        n->get_parameter_or(name, ans, alternative);
        return ans;
    }
    
    static Eigen::Vector3d readVector3d(rclcpp::Node::SharedPtr &n, const std::string& name)
    {
        std::vector<double> v = CommonUtils::readParam<std::vector<double>>(n, name);
        return Eigen::Vector3d(v[0], v[1], v[2]);
    }          

    static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
    {
        Eigen::Vector3d n = R.col(0);
        Eigen::Vector3d o = R.col(1);
        Eigen::Vector3d a = R.col(2);

        Eigen::Vector3d ypr(3);
        double y = atan2(n(1), n(0));
        double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
        double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
        ypr(0) = y;
        ypr(1) = p;
        ypr(2) = r;

        return ypr / M_PI * 180.0;
    }      

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr)
    {
        typedef typename Derived::Scalar Scalar_t;

        Scalar_t y = ypr(0) / 180.0 * M_PI;
        Scalar_t p = ypr(1) / 180.0 * M_PI;
        Scalar_t r = ypr(2) / 180.0 * M_PI;

        Eigen::Matrix<Scalar_t, 3, 3> Rz;
        Rz << cos(y), -sin(y), 0,
                sin(y), cos(y), 0,
                0, 0, 1;

        Eigen::Matrix<Scalar_t, 3, 3> Ry;
        Ry << cos(p), 0., sin(p),
                0., 1., 0.,
                -sin(p), 0., cos(p);

        Eigen::Matrix<Scalar_t, 3, 3> Rx;
        Rx << 1., 0., 0.,
                0., cos(r), -sin(r),
                0., sin(r), cos(r);

        return Rz * Ry * Rx;
    }

    static Eigen::Matrix3d g2R(const Eigen::Vector3d &g)
    {
        Eigen::Matrix3d R0;
        Eigen::Vector3d ng1 = g.normalized();
        Eigen::Vector3d ng2{0, 0, 1.0};
        R0 = Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        double yaw = CommonUtils::R2ypr(R0).x();
        R0 = CommonUtils::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
        return R0;
    }    

    static int64_t ms2ns(const float t_ms) {
        return (t_ms * float(1e6));
    }    

    static bool time_list(pcl::PointXYZINormal &x, pcl::PointXYZINormal &y) {return (x.intensity < y.intensity);};

    static geometry_msgs::msg::PoseStamped pose2msg(const int64_t t, const Eigen::Vector3d& pos,
                                              const Eigen::Quaterniond& orient)
    {
        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = rclcpp::Time(t);
        msg.pose.position.x = pos.x();
        msg.pose.position.y = pos.y();
        msg.pose.position.z = pos.z();
        msg.pose.orientation.w = orient.w();
        msg.pose.orientation.x = orient.x();
        msg.pose.orientation.y = orient.y();
        msg.pose.orientation.z = orient.z();
        return msg;
    }    

    static geometry_msgs::msg::Point32 getPointMsg(const Eigen::Vector3d& p)
    {
        geometry_msgs::msg::Point32 p_msg;
        p_msg.x = p.x();
        p_msg.y = p.y();
        p_msg.z = p.z();
        return p_msg;
    }    

    template<typename T>
    static bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const Eigen::aligned_vector<pcl::PointXYZINormal> &point, const T &threshold,  double &plane_cov, double cov_threshold)
    {
        // return false;
        Eigen::Matrix<T, 5, 3> A;
        Eigen::Matrix<T, 5, 1> b;
        Eigen::Matrix<T, 5, 5> W = Eigen::Matrix<T, 5, 5>::Identity();
        double cov_sum = 0;
        plane_cov = 0;

        A.setZero();
        b.setOnes();
        b *= -1.0f;
        for (int j = 0; j < 5; j++)
        {
            A(j,0) = point[j].x;
            A(j,1) = point[j].y;
            A(j,2) = point[j].z;
            W(j, j) = point[j].normal_x; // uncertainty in world map
            // RCLCPP_INFO(rclcpp::get_logger("esti_plane"), "point[%d]: %f, %f, %f, cov: %f", j, point[j].x, point[j].y, point[j].z, W(j, j));
            if (point[j].normal_x > cov_threshold)
            {
                return false;
            }
            cov_sum += abs(cov_threshold - W(j, j));
        }
        if ((W(0, 0) > 0.00001))
        {
            for (int j = 0; j < 5; j++)
            {
                plane_cov += ((cov_threshold - W(j, j)) / cov_sum) * ((cov_threshold - W(j, j)) / cov_sum) * W(j, j);
            }
        }
        // RCLCPP_INFO(rclcpp::get_logger("esti_plane"), "plane cov: %f", plane_cov);

        Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
        T n = normvec.norm();
        pca_result(0) = normvec(0) / n;
        pca_result(1) = normvec(1) / n;
        pca_result(2) = normvec(2) / n;
        pca_result(3) = 1.0 / n;
        for (int j = 0; j < 5; j++)
        {
            if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) > threshold)
            {
                return false;
            }
        }
        return true;
    }

    // for matrix debugging
    template <typename Derived>
    static std::string eigenToString(const Eigen::MatrixBase<Derived>& m)
    {
        // Eigen::IOFormat fmt = Eigen::IOFormat(3, 0, ", ", "\n", "[", "]");
        std::ostringstream ss;
            for (int i = 0; i < m.rows(); ++i) {
            ss << "[";
            for (int j = 0; j < m.cols(); ++j) {
                ss << m(i, j);
                if (j + 1 < m.cols()) ss << ", ";
            }
            ss << "]";
            if (i + 1 < m.rows()) ss << "\n";
        }

        return ss.str();
    }
};

struct LidarConfig {
    std::string topic;
    std::string type;
    int scan_line;
    float blind;
    float max_range;
    float min_z;
    Eigen::Quaterniond q_lb;
    Eigen::Vector3d t_lb;
    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;
    double w_pt;
    double w_vel;

    LidarConfig() = default;

    LidarConfig(rclcpp::Node::SharedPtr& nh, const std::string& prefix) {
        RCLCPP_INFO(nh->get_logger(), "Creating LidarConfig with prefix: \"%s\"", prefix.c_str());
        // std::cout << "Creating LidarConfig with prefix: \"" << prefix << "\"" << std::endl;
        topic = CommonUtils::readParam<std::string>(nh, prefix + "topic_lidar");
        type = CommonUtils::readParam<std::string>(nh, prefix + "lidar_type");
        scan_line = CommonUtils::readParam<int>(nh, prefix + "scan_line");
        blind = CommonUtils::readParam<float>(nh, prefix + "blind", 0.5f);
        max_range = CommonUtils::readParam<float>(nh, prefix + "max_range", 100.0f);
        min_z = CommonUtils::readParam<float>(nh, prefix + "min_z", -2.0f);
        // std::vector<double> q_lb_v = CommonUtils::readParam<std::vector<double>>(nh, prefix + "q_lb");
        // q_lb = Eigen::Quaterniond(q_lb_v.at(0), q_lb_v.at(1), q_lb_v.at(2), q_lb_v.at(3));
        // t_lb = CommonUtils::readVector3d(nh, prefix + "t_lb");
        std::vector<double> q_bl_v = CommonUtils::readParam<std::vector<double>>(nh, prefix + "q_bl");
        q_bl = Eigen::Quaterniond(q_bl_v.at(0), q_bl_v.at(1), q_bl_v.at(2), q_bl_v.at(3));
        t_bl = CommonUtils::readVector3d(nh, prefix + "t_bl");

        w_pt = CommonUtils::readParam<double>(nh, prefix + "w_pt", 0.01);
        w_vel = CommonUtils::readParam<double>(nh, prefix + "w_vel", 0.1);
    }
};

struct Parameters {
    Eigen::Vector3d cov_acc;
    Eigen::Vector3d cov_gyro;
    Eigen::Vector3d gravity;
    double nn_thresh;
    double coeff_cov;
    double cov_grav;

    Parameters() {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PointData {
    int64_t time_ns;
    pcl::PointXYZINormal pt; // pt.curvature: rcs
    Eigen::Vector3d pt_l; // assigned in constructor, local frame
    pcl::PointXYZINormal pt_w; // need to update
    Eigen::Matrix3d cov; // pointwise covariance in local frame
    Eigen::Vector3d normvec = Eigen::Vector3d::Zero();
    double dist; // associated plane : ax+by+cz+'dist'=0 :  (need to update)
    // Eigen::Vector3d vox_center;
    bool if_valid;
    bool is_plane = false; // if the point is a plane point

    Eigen::aligned_vector<pcl::PointXYZINormal> nearest_points;
    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;

    // measurement residual variables
    double zp = 0;
    Eigen::Matrix<double, 1, 24> H = Eigen::Matrix<double, 1, 24>::Zero();
    double zpl = 0;
    Eigen::Matrix<double, 1, 24> H_pl = Eigen::Matrix<double, 1, 24>::Zero();
    double z_dopp = 0;
    Eigen::Matrix<double, 1, 24> H_dopp = Eigen::Matrix<double, 1, 24>::Zero();
    double var_pt; // geo residual weight
    double var_vel; // doppler residual weight
    double pt_unc;
    double plane_cov;
    Eigen::Matrix<double, 1, 6> loc_contrib = Eigen::Matrix<double, 1, 6>::Zero(); // contribution of the point to the location constraint

    PointData() {};
    PointData(const pcl::PointXYZINormal& pt_in, int64_t fr_start_time, const Eigen::Quaterniond& q_bl_in,
        const Eigen::Vector3d& t_bl_in, double w_pt, double w_vel=0.1) : pt(pt_in), pt_l(Eigen::Vector3d(pt_in.x, pt_in.y, pt_in.z)), 
        q_bl(q_bl_in), t_bl(t_bl_in) {
        time_ns = fr_start_time + CommonUtils::ms2ns(pt_in.intensity);
        dist = 0;
        if_valid = false;
        is_plane = false;
        var_pt = w_pt;
        var_vel = w_vel;
        pt_unc = w_pt; // init
        plane_cov = 0.01; // init, can be updated later
        cov = Eigen::Matrix3d::Identity() * 0.1;
    }

    PointData(const PointData& other) :
        time_ns(other.time_ns),
        pt(other.pt), 
        pt_l(other.pt_l),
        pt_w(other.pt_w), 
        cov(other.cov), 
        normvec(other.normvec),
        dist(other.dist), 
        if_valid(other.if_valid),
        is_plane(other.is_plane),
        nearest_points(other.nearest_points), 
        q_bl(other.q_bl),
        t_bl(other.t_bl),
        zp(other.zp),
        H(other.H),
        zpl(other.zpl),
        H_pl(other.H_pl),
        z_dopp(other.z_dopp),
        H_dopp(other.H_dopp),
        var_pt(other.var_pt),
        var_vel(other.var_vel),
        pt_unc(other.pt_unc),
        plane_cov(other.plane_cov),
        loc_contrib(other.loc_contrib) {}

    PointData& operator=(const PointData& other) {
        if (this != &other) { 
            this->time_ns = other.time_ns;
            this->pt = other.pt;
            this->pt_l = other.pt_l;
            this->pt_w = other.pt_w;
            this->cov = other.cov;
            this->normvec = other.normvec;
            this->dist = other.dist;
            this->if_valid = other.if_valid;
            this->is_plane = other.is_plane;
            this->nearest_points = other.nearest_points;
            this->q_bl = other.q_bl;
            this->t_bl = other.t_bl;   
            this->zp = other.zp;
            this->H = other.H;
            this->zpl = other.zpl;
            this->H_pl = other.H_pl;
            this->z_dopp = other.z_dopp;
            this->H_dopp = other.H_dopp;
            this->var_pt = other.var_pt;         
            this->var_vel = other.var_vel;
            this->pt_unc = other.pt_unc;
            this->plane_cov = other.plane_cov;
            this->loc_contrib = other.loc_contrib;
        }
        return *this;
    }    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
