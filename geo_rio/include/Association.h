#pragma once

#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "utils/common_utils.h"
#include "SplineState.h"
#include "ikd-Tree/ikd_Tree.h"
#include <chrono>

inline float plane_thresh = 0.3;
class Association
{
public:

    // template<class PointType>
    // static void pointBodyToWorld(int64_t t_ns, const SplineState* spline, const PointType& pi, PointType& po, const Eigen::Vector3d& t_bl, const Eigen::Quaterniond& q_bl) 
    // {
    //     Eigen::Quaterniond q;
    //     Eigen::Vector3d pos = spline->itpPosition(t_ns);
    //     spline->itpQuaternion(t_ns, &q);
    //     Eigen::Vector3f p_lid(pi.x, pi.y, pi.z);
    //     Eigen::Vector3f p_global = q.cast<float>() * (q_bl.cast<float>() * p_lid + t_bl.cast<float>()) + pos.cast<float>();
    //     po.x = p_global(0);
    //     po.y = p_global(1);
    //     po.z = p_global(2);
    //     po.curvature = pi.curvature;
    // }

    static void pointLocalToWorld(int64_t t_ns, const SplineState* spline, const PointData& pi, PointData& po) 
    {   
        po = pi;
        Eigen::Quaterniond q;
        Eigen::Matrix3d pos_unc;
        Eigen::Vector3d pos = spline->itpPosition(t_ns, nullptr, &pos_unc);
        Eigen::Matrix3d rot_unc;
        spline->itpQuaternion(t_ns, &q, nullptr, nullptr, nullptr, &rot_unc);

        Eigen::Vector3d p_lid(pi.pt.x, pi.pt.y, pi.pt.z);
        Eigen::Vector3d p_imu = pi.q_bl * p_lid + pi.t_bl; // point in imu frame
        Eigen::Vector3d p_global = q * p_imu + pos;
        po.pt_w.x = p_global(0);
        po.pt_w.y = p_global(1);
        po.pt_w.z = p_global(2);
        po.pt_w.curvature = pi.pt.curvature; // rcs

        Eigen::Matrix3d skew_sym = Eigen::Matrix3d::Zero();
        skew_sym <<  0, -p_imu(2), p_imu(1),
                    p_imu(2), 0, -p_imu(0),
                    -p_imu(1), p_imu(0), 0;

        // (25), (26): consolidate measurement uncertainty with the propagated pose uncertainty
        Eigen::Matrix3d R = q.toRotationMatrix();
        Eigen::Matrix3d cov_i = (pi.q_bl).toRotationMatrix() * pi.cov * (pi.q_bl).toRotationMatrix().transpose();
        Eigen::Matrix3d cov_w = R * cov_i * R.transpose()
                                + R * skew_sym * rot_unc * skew_sym.transpose() * R.transpose()
                                + pos_unc; // uncertainty in world frame
        po.pt_w.normal_x = cov_w(0, 0) + cov_w(1, 1) + cov_w(2, 2); // uncertainty weight
    }

    static void findCorresp(int& effect_num_k, const SplineState* spline, KD_TREE<pcl::PointXYZINormal>* ikdtree, Eigen::aligned_deque<PointData>& pt_meas)
    {
        int num_pt = pt_meas.size();
        int plane_cnt = 0;
        int ptp_cnt = 0;

        // TODO : parallelize this part
        #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic) shared(pt_meas, spline, ikdtree, num_pt, effect_num_k, plane_cnt, ptp_cnt)
        for (int i = 0; i < num_pt; i++) {
            PointData& pt_data = pt_meas[i];
            pt_data.if_valid = false;
            pt_data.is_plane = false;
            if (!(spline->numKnots() == 4) && !(pt_data.time_ns <= spline->maxTimeNs() && pt_data.time_ns >= spline->maxTimeNs() - 4*spline->getKnotTimeIntervalNs())) {                           
                continue;
            }
            // Association::pointBodyToWorld(pt_data.time_ns, spline, pt_data.pt, pt_data.pt_w, pt_data.t_bl, pt_data.q_bl);
            Association::pointLocalToWorld(pt_data.time_ns, spline, pt_data, pt_data);

            std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
            pt_data.nearest_points.clear();
            ikdtree->Nearest_Search(pt_data.pt_w, NUM_MATCH_POINTS, pt_data.nearest_points, pointSearchSqDis, INFINITY); // sqrt(5) = 2.236
            if (pt_data.nearest_points.size() >= (size_t)NUM_MATCH_POINTS && pointSearchSqDis[NUM_MATCH_POINTS - 1] < 5) {     
                Eigen::Vector4f pabcd;       
                pabcd.setZero();
                pt_data.if_valid = true;
                
                double plane_cov = 0.0;
                const double cov_threshold = 1.5;

                // RCLCPP_INFO(rclcpp::get_logger("Association"), "*****************************************");
                if (CommonUtils::esti_plane(pabcd, pt_data.nearest_points, plane_thresh, plane_cov, cov_threshold)) {
                // if (CommonUtils::esti_plane(pabcd, pt_data.nearest_points, 0.3f, plane_cov, cov_threshold)) {
                    float pd2 = pabcd(0) * pt_data.pt_w.x + pabcd(1) * pt_data.pt_w.y + pabcd(2) * pt_data.pt_w.z + pabcd(3);
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(pt_data.pt_l.norm());

                    if (s > 0.1) {
                        pt_data.is_plane = true;
                        pt_data.normvec = Eigen::Vector3d(pabcd[0], pabcd[1], pabcd[2]);
                        pt_data.dist = pabcd(3);
                        pt_data.plane_cov = plane_cov;
                        #pragma omp atomic
                        plane_cnt ++;
                    }
                    else{
                        // plane is too far : ptp
                        // pt_data.normvec = pt_data.nearest_points[0].getVector3fMap().cast<double>();
                        #pragma omp atomic
                        ptp_cnt ++;
                    }
                }
                else{
                    // not a plane : ptp
                    // pt_data.normvec = pt_data.nearest_points[0].getVector3fMap().cast<double>();
                    #pragma omp atomic
                    ptp_cnt ++;
                }
            }
            else{
                // not enough neighbors : don't use this point
            }
        }

        for (int i = 0; i < num_pt; i++) {
            if (pt_meas[i].if_valid) {
                effect_num_k++;
            } 
        }
    } 
};