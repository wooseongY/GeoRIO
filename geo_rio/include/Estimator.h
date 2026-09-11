#pragma once

#include "SplineState.h"
#include "Association.h"
#include "voxelmapper/voxelmapper.h"

template<int XSIZE>
class Estimator
{
  public:
    static const int CP_SIZE = 24;
    static const int BA_OFFSET = 24;
    static const int BG_OFFSET = 27;  
    int n_iter = 1;

    double min_cov_thresh = 0.01; // minimum uncertainty threshold
    double max_cov_thresh = 0.1; // maximum uncertainty threshold, 0.5?
    double min_cov_bound = 1;
    double max_cov_bound = 3;

    double min_plane_cov_thresh = 0.005; // minimum uncertainty threshold
    double max_plane_cov_thresh = 0.01; // maximum uncertainty threshold
    double min_plane_cov_bound = 0.0075;
    double max_plane_cov_bound = 0.0125;

    Estimator() {};

    void setState(int64_t dt_ns, int64_t start_t_ns, const Eigen::Vector3d& t0, const Eigen::Quaterniond& q0, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& Q, const Eigen::Matrix<double, XSIZE, XSIZE>& P)
    {
        spl.init(dt_ns, 0, start_t_ns, 0, t0, q0);
        for (int i = 0; i < 4; i++) {
            spl.addOneStateKnot(t0, Eigen::Vector3d::Zero());
        }        
        cov_sys = Q;
        cov_rcp = P;
        a_mat = Eigen::Matrix<double, XSIZE, XSIZE>::Zero();
        Eigen::Matrix<double, 6, 6> matblock = Eigen::Matrix<double, 6, 6>::Zero();
        matblock.topLeftCorner<3, 3>().setIdentity();
        matblock.bottomRightCorner<3, 3>().setIdentity();

        a_mat.block(0, 6, 6, 6) = matblock;
        a_mat.block(6, 12, 6, 6) = matblock;
        a_mat.block(12, 18, 6, 6) = matblock;
        a_mat.block(18, 0, 3, 3) = - Eigen::Matrix3d::Identity();
        a_mat.block(18, 12, 3, 3) = 2 * Eigen::Matrix3d::Identity();
        a_mat.block(21, 9, 3, 3) = Eigen::Matrix3d::Identity();

        if constexpr (XSIZE == 30) {
            a_mat.block(BA_OFFSET, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
            a_mat.block(BG_OFFSET, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
        }
        spl.updateRCPCovs(cov_rcp.template topLeftCorner<24, 24>());
    }

    Eigen::Matrix<double, XSIZE, 1> getState()
    {
        Eigen::Matrix<double, CP_SIZE, 1> cps_win = spl.getRCPs();
        Eigen::Matrix<double, XSIZE, 1> state;
        if constexpr (XSIZE == 24) {
            state = cps_win;
        } else {
            state << cps_win, ba, bg;
        }
        return state;
    }

    void updateIEKFLiDAR(Eigen::aligned_deque<PointData>& pt_meas, KD_TREE<pcl::PointXYZINormal>* ikdtree, const double pt_thresh, const double cov_thresh)
    {
        const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = cov_rcp;
        Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
        bool converged = true;
        int num_tot_eff = 0;
        int t = 0;
        for (int i = 0; i < max_iter; i++) {
            Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
            if (converged) {
                num_tot_eff = 0;
                Association::findCorresp(num_tot_eff, &spl, ikdtree, pt_meas);
            }
            if (num_tot_eff > 0) {
                updateLiDAR(pt_meas, num_tot_eff, rcp_prop, cov_prop, pt_thresh, cov_thresh);
            } else {
                break;
            }
            converged = true;
            Eigen::Matrix<double, XSIZE, 1> state_af = getState();
            if ((state_af - rcpi).norm() > eps) {
                converged = false;
            } else {
                t++;
            }
            if(!t && i == max_iter - 2) {
                converged = true;
            }
            if ((t > n_iter) || (i == max_iter - 1)) {
                cov_rcp = ( Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;
                cov_rcp = 0.5 * (cov_rcp + cov_rcp.transpose());

                spl.updateRCPCovs(cov_rcp.template topLeftCorner<24, 24>());
                break;
            }  
        }
    }

    // kdtree
    void updateIEKFLiDARInertial(Eigen::aligned_deque<PointData>& pt_meas, KD_TREE<pcl::PointXYZINormal>* ikdtree, const double pt_thresh,
        Eigen::aligned_deque<ImuData>& imu_meas, const Eigen::Vector3d& g, const Eigen::Vector3d& cov_acc, const Eigen::Vector3d& cov_gyro, const double cov_thresh, const double cov_grav)
    {
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "kdtree, updateIEKFLiDARInertial with %zu imu measurements, %zu lidar points", imu_meas.size(), pt_meas.size());
        const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = cov_rcp;
        Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
        bool converged = true;
        int num_tot_eff = 0;
        int t = 0;
        auto duration_asso = 0;
        auto duration_upd = 0;
        for (int i = 0; i < max_iter; i++) {
            Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
            if (converged) {
                num_tot_eff = 0;
                // auto T1 = std::chrono::steady_clock::now();
                Association::findCorresp(num_tot_eff, &spl, ikdtree, pt_meas);
                // auto T2 = std::chrono::steady_clock::now();
                // duration_asso += std::chrono::duration_cast<std::chrono::microseconds>(T2 - T1).count();
            }
            if (num_tot_eff > 0 && imu_meas.empty()) {
                updateLiDAR(pt_meas, num_tot_eff, rcp_prop, cov_prop, pt_thresh, cov_thresh);
            } else if (num_tot_eff > 0) {
                // auto T1 = std::chrono::steady_clock::now();
                updateLiDARInertial(pt_meas, imu_meas, num_tot_eff, rcp_prop, cov_prop, pt_thresh, cov_thresh, g, cov_acc, cov_gyro, cov_grav);
                // auto T2 = std::chrono::steady_clock::now();
                // duration_upd += std::chrono::duration_cast<std::chrono::microseconds>(T2 - T1).count();
                
            } else {
                break;
            }
            converged = true;
            Eigen::Matrix<double, XSIZE, 1> state_af = getState();
            if ((state_af - rcpi).norm() > eps) {
                converged = false;
            } else {
                t++;
            }
            if(!t && i == max_iter - 2) {
                converged = true;
            }
            if ((t > n_iter) || (i == max_iter - 1)) {

                cov_rcp = (Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;
                
                cov_rcp = 0.5 * (cov_rcp + cov_rcp.transpose());
                spl.updateRCPCovs(cov_rcp.template topLeftCorner<24, 24>());
                break;
            }
        }
        // std::ofstream time_file("time.txt", std::ios::app);
        // time_file.setf(std::ios::fixed, std::ios::floatfield);
        // time_file.precision(9);
        // time_file << duration_asso << "," << duration_upd << " ";
        // time_file.close();
    }

    // voxel map
    // void updateIEKFLiDARInertial(Eigen::aligned_deque<PointData>& pt_meas, VoxelMapManagerPtr voxelmap_manager, const double pt_thresh,
    //     Eigen::aligned_deque<ImuData>& imu_meas, const Eigen::Vector3d& g, const Eigen::Vector3d& cov_acc, const Eigen::Vector3d& cov_gyro, const double cov_thresh, const double cov_grav)
    // {
    //     RCLCPP_INFO(rclcpp::get_logger("Estimator"), "voxel map, updateIEKFLiDARInertial with %zu imu measurements, %zu lidar points", imu_meas.size(), pt_meas.size());
    //     const Eigen::Matrix<double, XSIZE, XSIZE> cov_prop = cov_rcp;
    //     Eigen::Matrix<double, XSIZE, 1> rcp_prop = getState();
    //     bool converged = true;
    //     int num_tot_eff = 0;
    //     int t = 0;
    //     for (int i = 0; i < max_iter; i++) {
    //         Eigen::Matrix<double, XSIZE, 1> rcpi = getState();
    //         if (converged) {
    //             num_tot_eff = 0;
    //             // Association::findCorresp(num_tot_eff, &spl, ikdtree, pt_meas);
                
    //             // TODO: find correspondence in voxel map
    //             for(auto& pt : pt_meas) {
    //                 Association::pointLocalToWorld(pt.time_ns, &spl, pt, pt);
    //             }
    //             voxelmap_manager->findCorrespondVoxel(num_tot_eff, pt_meas);
    //         }
    //         RCLCPP_INFO(rclcpp::get_logger("Estimator"), "Find correspondence, num_tot_eff: %d", num_tot_eff);
    //         if (num_tot_eff > 0 && imu_meas.empty()) {
    //             updateLiDAR(pt_meas, num_tot_eff, rcp_prop, cov_prop, pt_thresh, cov_thresh);
    //         } else if (num_tot_eff > 0) {
    //             updateLiDARInertial(pt_meas, imu_meas, num_tot_eff, rcp_prop, cov_prop, pt_thresh, cov_thresh, g, cov_acc, cov_gyro, cov_grav);
    //         } else {
    //             break;
    //         }
    //         converged = true;
    //         Eigen::Matrix<double, XSIZE, 1> state_af = getState();
    //         if ((state_af - rcpi).norm() > eps) {
    //             converged = false;
    //         } else {
    //             t++;
    //         }
    //         if(!t && i == max_iter - 2) {
    //             converged = true;
    //         }
    //         if ((t > n_iter) || (i == max_iter - 1)) {
    //             cov_rcp = (Eigen::MatrixXd::Identity(XSIZE, XSIZE) - KH) * cov_prop;

    //             RCLCPP_INFO(rclcpp::get_logger("Estimator"), "bef covariance:\n%s", CommonUtils::eigenToString(cov_rcp).c_str());

    //             const auto& C = constraint_mat;
    //             Eigen::MatrixXd B = C.rightCols(6); // k×6
    //             Eigen::MatrixXd BBt = B * B.transpose(); // k×k
    //             Eigen::MatrixXd BBt_inv = BBt.ldlt().solve(Eigen::MatrixXd::Identity(B.rows(), B.rows())); // (B B^T)^(-1)
    //             Eigen::MatrixXd P_B = Eigen::MatrixXd::Identity(6, 6) - B.transpose() * BBt_inv * B; // 6×6 projection matrix

    //             // Full projection matrix
    //             Eigen::Matrix<double, XSIZE, XSIZE> P_C = Eigen::Matrix<double, XSIZE, XSIZE>::Identity();
    //             P_C.template block<6, 6>(18, 18) = P_B; // Apply to indices 18–23 (i=3)

    //             RCLCPP_INFO(rclcpp::get_logger("Estimator"), "Constrained covariance, P_C:\n%s", CommonUtils::eigenToString(P_C).c_str());

    //             // Constrained covariance
    //             // cov_rcp = P_C * cov_rcp;
    //             RCLCPP_INFO(rclcpp::get_logger("Estimator"), "aft covariance:\n%s", CommonUtils::eigenToString(P_C * cov_rcp).c_str());

    //             cov_rcp = 0.5 * (cov_rcp + cov_rcp.transpose());
    //             break;
    //         }
    //     }
    // }

    void propRCP(int64_t t)
    {
        if (spl.maxTimeNs() >= t) {
            cov_rcp += cov_sys;
        } else {
            while (spl.maxTimeNs() < t) {
                Eigen::Matrix<double, 24, 1> cps_win = spl.getRCPs();
                Eigen::Vector3d cp_prop_pos = 2 * cps_win.segment<3>(12) - cps_win.segment<3>(0); // 03 69 1215 1821
                Eigen::Vector3d delta = cps_win.segment<3>(9); // change with a_mat
                spl.addOneStateKnot(cp_prop_pos, delta);
                cov_rcp = a_mat * cov_rcp * a_mat.transpose() + cov_sys;
            }
        }
        // Push the propagated covariance into the spline so (19) and (22) see the prior.
        spl.updateRCPCovs(cov_rcp.template topLeftCorner<24, 24>());
    }

    SplineState* getSpline() {
        return &spl;
    }     

  private:
    SplineState spl;
    Eigen::Matrix<double, XSIZE, XSIZE> cov_rcp;  
    Eigen::Matrix<double, XSIZE, XSIZE> cov_sys; 
    Eigen::Matrix<double, XSIZE, XSIZE> a_mat;   
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    Eigen::Vector3d ba = Eigen::Vector3d::Zero();     
    Eigen::Matrix<double, XSIZE, XSIZE> KH;
    Eigen::MatrixXd constraint_mat;
    int max_iter = 5;
    double eps = 0.001;
    int valid_cnt = 0;
    int plane_cnt = 0;

    void prepIMU(ImuData& imu_data, const Eigen::Vector3d& gw)
    {
        Eigen::Quaterniond q_itp;
        Eigen::Vector3d rot_vel;
        Jacobian43 J_ortdel;
        Jacobian J_line_acc;
        Jacobian33 J_gyro;
        Eigen::Matrix3d rot_unc, omg_unc;
        spl.itpQuaternion(imu_data.time_ns, &q_itp, &rot_vel, &J_ortdel, &J_gyro, &rot_unc, &omg_unc);
        Eigen::Matrix3d acc_unc;
        Eigen::Vector3d a_w_no_g = spl.itpPosition<2>(imu_data.time_ns, &J_line_acc, &acc_unc);
        Eigen::Vector3d a_w = a_w_no_g + gw;
        Eigen::Matrix3d RT = q_itp.toRotationMatrix().transpose();   
        Eigen::Matrix<double, 3, 4> drot;
        Quater::drot(a_w, q_itp, drot);
        Eigen::Matrix<double, 6, XSIZE> Hi = Eigen::Matrix<double, 6, XSIZE>::Zero();
        int RCP_st_id = spl.numKnots() - 4;
        for (int i = 0; i < (int) J_line_acc.d_val_d_knot.size(); i++) {
            int j = J_line_acc.start_idx + i - RCP_st_id;
            if (j >= 0) {
                Hi.block(0, j*6, 3, 3) = RT * J_line_acc.d_val_d_knot[i];
                Hi.block(0, j*6 + 3, 3, 3) = drot * J_ortdel.d_val_d_knot[i];
                Hi.block(3, j*6 + 3, 3, 3) = J_gyro.d_val_d_knot[i];                
            }
        }       
        Hi.block(0, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
        Hi.block(3, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity(); // these two not used          
        imu_data.imu_itp.head<3>() = RT * a_w + ba;
        imu_data.imu_itp.tail<3>() = rot_vel + bg;
        imu_data.H = Hi.template leftCols<24>();


        Eigen::Matrix3d R = q_itp.toRotationMatrix();
        Eigen::Matrix<double, 1, XSIZE> Hi_g = Eigen::Matrix<double, 1, XSIZE>::Zero();

        Eigen::Vector3d g_norm = gw / gw.norm(); // gravity norm
        Eigen::Vector3d grav = R * (imu_data.accel - ba) - a_w_no_g; // estimation
        Eigen::Vector3d grav_norm = grav / grav.norm();

        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), ">>>>>IMU meas<<<<<");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "a_I: %f, %f, %f", (RT * a_w + ba)(0), (RT * a_w + ba)(1), (RT * a_w + ba)(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "imu_data.accel: %f, %f, %f", imu_data.accel(0), imu_data.accel(1), imu_data.accel(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "rot vel: %f, %f, %f", rot_vel(0), rot_vel(1), rot_vel(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "imu_data.gyro: %f, %f, %f", imu_data.gyro(0), imu_data.gyro(1), imu_data.gyro(2));

        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "ba: %f, %f, %f", ba(0), ba(1), ba(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "bg: %f, %f, %f", bg(0), bg(1), bg(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "gw: %f, %f, %f", gw(0), gw(1), gw(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "g_norm: %f, %f, %f", g_norm(0), g_norm(1), g_norm(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "grav: %f, %f, %f", grav(0), grav(1), grav(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "grav_norm: %f, %f, %f", grav_norm(0), grav_norm(1), grav_norm(2));
        
        Eigen::Matrix3d dn_du = (Eigen::Matrix3d::Identity() - grav_norm * grav_norm.transpose()) / grav.norm();
        Quater::drotInv((imu_data.accel - ba), q_itp, drot);
        imu_data.z_grav = 0.0;
        imu_data.H_grav.setZero();
        if (grav.norm() > 1e-6 && grav_norm(2) > 0.9){
            imu_data.z_grav = 1 - g_norm.dot(grav_norm);

            for (int i = 0; i < (int) J_line_acc.d_val_d_knot.size(); i++) {
                int j = J_line_acc.start_idx + i - RCP_st_id;
                if (j >= 0) {
                    Hi_g.block(0, j*6, 1, 3) = g_norm.transpose() * dn_du * J_line_acc.d_val_d_knot[i];
                    Hi_g.block(0, j*6 + 3, 1, 3) = - g_norm.transpose() * dn_du * drot * J_ortdel.d_val_d_knot[i];
                }
            }    
            Hi_g.block(0, BA_OFFSET, 1, 3) = g_norm.transpose() * dn_du * R;
            imu_data.H_grav = Hi_g.template leftCols<XSIZE>();
            // imu_data.H_grav(0, 23) = 0.0;
        }
        
        // TODO : estimate omega uncertainty 
        double omega_unc = omg_unc(0, 0) + omg_unc(1, 1) + omg_unc(2, 2);
        double accel_unc = acc_unc(0, 0) + acc_unc(1, 1) + acc_unc(2, 2);
        imu_data.omega_unc = omega_unc;
        imu_data.acc_unc = accel_unc;
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "imu a_bod: %f, %f, %f", imu_data.imu_itp(0), imu_data.imu_itp(1), imu_data.imu_itp(2));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "imu rot_vel: %f, %f, %f", imu_data.imu_itp(3), imu_data.imu_itp(4), imu_data.imu_itp(5));
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "omega_unc: %f, acc_unc: %f", omega_unc, acc_unc);
    }

    Eigen::Matrix3d sortEigenvectorsToAxes(const Eigen::Matrix3d &V) const
    {
        // Define the three coordinate axes
        const Eigen::Vector3d axes[3] = {
            Eigen::Vector3d::UnitX(),
            Eigen::Vector3d::UnitY(),
            Eigen::Vector3d::UnitZ()
        };
    
        // Compute "closeness" between each axis and each eigenvector
        double closeness[3][3];
        for (int vec_idx = 0; vec_idx < 3; ++vec_idx) {
            for (int axis_idx = 0; axis_idx < 3; ++axis_idx) {
                closeness[axis_idx][vec_idx] = 
                    std::abs(axes[axis_idx].dot(V.col(vec_idx)));
            }
        }
    
        // Test all 6 permutations of mapping vectors->axes and pick the best
        std::array<int, 3> bestPerm = {0, 1, 2};
        std::array<int, 3> perm = {0, 1, 2};
        double bestScore = -1.0;
        do {
            double score = 
                closeness[0][perm[0]] +
                closeness[1][perm[1]] +
                closeness[2][perm[2]];
            if (score > bestScore) {
                bestScore = score;
                bestPerm = perm;
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
    
        // Construct the sorted eigenvector matrix
        Eigen::Matrix3d V_sorted;
        for (int axis_idx = 0; axis_idx < 3; ++axis_idx) {
            V_sorted.col(axis_idx) = V.col(bestPerm[axis_idx]);
        }
        return V_sorted;
    }

    void prepLiDAR(PointData& pt_data) const    
    {
        if (!pt_data.if_valid)
            return;
        if (pt_data.is_plane) {
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "planar point");
            Eigen::Matrix<double, 1, XSIZE> Hi = Eigen::Matrix<double, 1, XSIZE>::Zero();
            Eigen::Quaterniond q_itp;
            Jacobian43 J_ortdel;
            Jacobian J_pos;
            Eigen::Matrix3d rot_unc;
            spl.itpQuaternion(pt_data.time_ns, &q_itp, nullptr, &J_ortdel, nullptr, &rot_unc);
            Eigen::Matrix3d pos_unc;
            Eigen::Vector3d p_itp = spl.itpPosition(pt_data.time_ns, &J_pos, &pos_unc);
            Eigen::Matrix3d R_IL = pt_data.q_bl.toRotationMatrix();
            Eigen::Vector3d pt_i = R_IL * pt_data.pt_l + pt_data.t_bl;
            Eigen::Vector3d pt_w = q_itp * pt_i + p_itp;

            pt_data.zpl = pt_data.normvec.dot(pt_w) + pt_data.dist; // point to plane

            Eigen::Matrix<double, 3, 4> drot;
            Quater::drotInv(pt_i, q_itp, drot);
            // Eigen::Matrix<double, 1, 4> tmp = pt_data.normvec.transpose();
            int RCP_st_id = spl.numKnots() - 4;
            for (int i = 0; i < (int) J_pos.d_val_d_knot.size(); i++) {
                int j = (int) J_pos.start_idx + i - RCP_st_id;
                if (j >= 0) {
                    Hi.block(0, j*6, 1, 3) = pt_data.normvec.transpose() * J_pos.d_val_d_knot[i];
                    Hi.block(0, j*6 + 3, 1, 3) = pt_data.normvec.transpose() * drot * J_ortdel.d_val_d_knot[i];
                }
            }  
            pt_data.H_pl = Hi.template leftCols<24>();

            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "[for point %f, %f, %f]", pt_data.pt_l(0), pt_data.pt_l(1), pt_data.pt_l(2));
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "planar residual: %f", pt_data.zp);

            // uncertainty
            Eigen::Matrix<double, 3, 3> skew_pt_i = Eigen::Matrix3d::Zero();
            skew_pt_i << 0, -pt_i(2), pt_i(1),
                        pt_i(2), 0, -pt_i(0),
                        -pt_i(1), pt_i(0), 0;


            // (25), (26): consolidate measurement uncertainty with the propagated pose uncertainty
            Eigen::Matrix3d R_itp = q_itp.toRotationMatrix();
            Eigen::Matrix3d cov_w = R_itp * R_IL * pt_data.cov * R_IL.transpose() * R_itp.transpose()
                                    + R_itp * skew_pt_i * rot_unc * skew_pt_i.transpose() * R_itp.transpose()
                                    + pos_unc; // uncertainty in world frame
            pt_data.pt_unc = cov_w(0, 0) + cov_w(1, 1) + cov_w(2, 2);

            Eigen::Matrix<double, 1, 6> loc_contrib = Eigen::Matrix<double, 1, 6>::Zero();
            loc_contrib.segment<3>(0) = pt_data.pt_l;
            // loc_contrib.segment<3>(0) = pt_data.normvec;
            loc_contrib.segment<3>(3) = pt_data.pt_l.cross(pt_data.normvec);
            Eigen::Matrix<double, 6, 6> pt_hess = loc_contrib.transpose() * loc_contrib;

            // compute information for each points
            Eigen::Matrix3d A_trans = pt_hess.block<3, 3>(0, 0);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_trans(A_trans);
            Eigen::Matrix3d eigen_vec_trans = sortEigenvectorsToAxes(es_trans.eigenvectors());

            Eigen::Vector3d trans_info = pt_data.normvec.transpose() * eigen_vec_trans;
            trans_info = trans_info.cwiseAbs();

            Eigen::Matrix3d A_rot = pt_hess.block<3, 3>(3, 3);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_rot(A_rot);
            Eigen::Matrix3d eigen_vec_rot = sortEigenvectorsToAxes(es_rot.eigenvectors());

            Eigen::Vector3d rot_info = (pt_data.pt_l.cross(pt_data.normvec)).normalized().transpose() * eigen_vec_rot;
            rot_info = rot_info.cwiseAbs();

            pt_data.loc_contrib.segment<3>(0) = trans_info;
            pt_data.loc_contrib.segment<3>(3) = rot_info;
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "point localizability information");
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "trans eigen vector: \n%s", CommonUtils::eigenToString(eigen_vec_trans).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "normal vector: \n%s", CommonUtils::eigenToString((pt_data.normvec).transpose()).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "rot eigen vector: \n%s", CommonUtils::eigenToString(eigen_vec_rot).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "pt_l x normvec: \n%s", CommonUtils::eigenToString((pt_data.pt_l.normalized().cross(pt_data.normvec)).transpose()).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "trans : %.9f, %.9f, %.9f", trans_info(0), trans_info(1), trans_info(2));
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "rot   : %.9f, %.9f, %.9f", rot_info(0), rot_info(1), rot_info(2));

            Eigen::Matrix<double, 24, 24> hess = pt_data.H.transpose() * pt_data.H;

            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "point body unc\n%s",
            //             CommonUtils::eigenToString(pt_data.cov).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "pos_unc\n%s",
            //             CommonUtils::eigenToString(pos_unc).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "rot_unc\n%s",
            //             CommonUtils::eigenToString(rot_unc).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "point world unc\n%s",
            //             CommonUtils::eigenToString(cov_w).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "pt_unc: %f", pt_data.pt_unc);
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "plane_cov: %f", pt_data.plane_cov);
        }
        else{
            // RCS weighted gaussian matching
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "----non planar point----");
            Eigen::Matrix<double, 1, XSIZE> Hi = Eigen::Matrix<double, 1, XSIZE>::Zero();
            Eigen::Quaterniond q_itp;
            Jacobian43 J_ortdel;
            Jacobian J_pos;
            Eigen::Matrix3d rot_unc;
            spl.itpQuaternion(pt_data.time_ns, &q_itp, nullptr, &J_ortdel, nullptr, &rot_unc);
            Eigen::Matrix3d pos_unc;
            Eigen::Vector3d p_itp = spl.itpPosition(pt_data.time_ns, &J_pos, &pos_unc);
            Eigen::Matrix3d R_IL = pt_data.q_bl.toRotationMatrix();
            Eigen::Vector3d pt_i = R_IL * pt_data.pt_l + pt_data.t_bl;
            Eigen::Vector3d pt_w = q_itp * pt_i + p_itp;
            
            size_t n_nn = pt_data.nearest_points.size();
            Eigen::Vector3d pt_nn_mean = Eigen::Vector3d::Zero();
            double rcs_sum = 0.0;
            for (size_t i = 0; i < n_nn; i++) {
                pt_nn_mean += pt_data.nearest_points[i].getVector3fMap().cast<double>() * pt_data.nearest_points[i].curvature; // RCS weighted mean
                rcs_sum += pt_data.nearest_points[i].curvature;
            }
            if (std::abs(rcs_sum) > 1e-9) {
                pt_nn_mean /= rcs_sum;
            } else {
                pt_nn_mean.setZero();
                for (size_t i = 0; i < n_nn; i++) {
                    pt_nn_mean += pt_data.nearest_points[i].getVector3fMap().cast<double>();
                }
                pt_nn_mean /= double(n_nn);
            }

            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "pt_w: %f, %f, %f", pt_w(0), pt_w(1), pt_w(2));
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "pt_nn_mean: %f, %f, %f", pt_nn_mean(0), pt_nn_mean(1), pt_nn_mean(2));
            
            pt_data.zp = (pt_w - pt_nn_mean).norm();
            if (pt_data.zp < 1e-6) {
                pt_data.zp = 0.0;
                pt_data.H.setZero();
                pt_data.pt_unc = pt_data.var_pt;
                pt_data.loc_contrib.setZero();
                return;
            }

            Eigen::Matrix<double, 3, 4> drot;
            Quater::drotInv(pt_i, q_itp, drot);
            // Eigen::Matrix<double, 1, 4> tmp = (pt_w - pt_nn_mean).transpose() * drot;
            int RCP_st_id = spl.numKnots() - 4;
            for (int i = 0; i < (int) J_pos.d_val_d_knot.size(); i++) {
                int j = (int) J_pos.start_idx + i - RCP_st_id;
                if (j >= 0) {
                    Hi.block(0, j*6, 1, 3) = (pt_w - pt_nn_mean).transpose() * J_pos.d_val_d_knot[i];
                    Hi.block(0, j*6 + 3, 1, 3) = (pt_w - pt_nn_mean).transpose() * drot * J_ortdel.d_val_d_knot[i];
                }
            }  
            pt_data.H = Hi.template leftCols<24>() / pt_data.zp;

            // (31): w_rcs = 1 / |rcs_p - rcs_mu|
            const double rcs_diff = std::abs(pt_data.pt_w.curvature - rcs_sum / double(n_nn));
            if (rcs_diff > 1.0) {
                pt_data.zp /= rcs_diff;
                pt_data.H /= rcs_diff;
            }

            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "[[[for point %f, %f, %f]]]", pt_w(0), pt_w(1), pt_w(2));
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "ptp residual: %f", pt_data.zp);
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "weight: %f", double(valid_cnt - plane_cnt) / valid_cnt);

            // uncertainty
            Eigen::Matrix<double, 3, 3> skew_pt_i = Eigen::Matrix3d::Zero();
            skew_pt_i << 0, -pt_i(2), pt_i(1),
                        pt_i(2), 0, -pt_i(0),
                        -pt_i(1), pt_i(0), 0;
            

            // (25), (26): consolidate measurement uncertainty with the propagated pose uncertainty
            Eigen::Matrix3d R_itp = q_itp.toRotationMatrix();
            Eigen::Matrix3d cov_w = R_itp * R_IL * pt_data.cov * R_IL.transpose() * R_itp.transpose()
                                    + R_itp * skew_pt_i * rot_unc * skew_pt_i.transpose() * R_itp.transpose()
                                    + pos_unc; // uncertainty in world frame
            pt_data.pt_unc = cov_w(0, 0) + cov_w(1, 1) + cov_w(2, 2);

            Eigen::Matrix<double, 1, 6> loc_contrib = Eigen::Matrix<double, 1, 6>::Zero();
            loc_contrib.segment<3>(0) = pt_data.pt_l;
            loc_contrib.segment<3>(3) = pt_data.pt_l.cross(pt_data.normvec);

            Eigen::Matrix<double, 24, 24> hess = pt_data.H.transpose() * pt_data.H;

            // auto fmt = Eigen::IOFormat(3, 0, ", ", "\n", "[", "]");
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "point body unc\n%s",
            //             CommonUtils::eigenToString(pt_data.cov).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "pos_unc\n%s",
            //             CommonUtils::eigenToString(pos_unc).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "rot_unc\n%s",
            //             CommonUtils::eigenToString(rot_unc).c_str());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "point world unc\n%s",
            //             CommonUtils::eigenToString(cov_w).c_str());

            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "pt_unc: %f", pt_data.pt_unc);
        }
    }

    void prepDoppler(PointData& pt_data) const    
    {
        if (pt_data.pt.normal_y == 1) {
            pt_data.pt.normal_z = 0.0; // ignore doppler if zero velocity
        }
        else { 
            Eigen::Matrix<double, 1, XSIZE> Hi = Eigen::Matrix<double, 1, XSIZE>::Zero();
            Eigen::Quaterniond q_itp;
            Eigen::Vector3d rot_vel;
            Jacobian J_vel;
            Jacobian43 J_ortdel;
            Jacobian J_pos;
            Jacobian33 J_gyro;
            spl.itpQuaternion(pt_data.time_ns, &q_itp, &rot_vel, &J_ortdel, &J_gyro);
            Eigen::Vector3d p_itp = spl.itpPosition(pt_data.time_ns, &J_pos);
            Eigen::Vector3d v_itp = spl.itpPosition<1>(pt_data.time_ns, &J_vel);
            Eigen::Matrix3d R_IL = pt_data.q_bl.toRotationMatrix();
            Eigen::Matrix3d RT = q_itp.toRotationMatrix().transpose();   

            // RCLCPP_INFO(rclcpp::get_logger("Estimator_dopp"), "q_itp: %f, %f, %f, %f", q_itp.x(), q_itp.y(), q_itp.z(), q_itp.w());
            // RCLCPP_INFO(rclcpp::get_logger("Estimator_dopp"), "p_itp: %f, %f, %f", p_itp(0), p_itp(1), p_itp(2));
            // RCLCPP_INFO(rclcpp::get_logger("Estimator_dopp"), "v_itp: %f, %f, %f", v_itp(0), v_itp(1), v_itp(2));
            // RCLCPP_INFO(rclcpp::get_logger("Estimator_dopp"), "rot_vel: %f, %f, %f", rot_vel(0), rot_vel(1), rot_vel(2));
            
            Eigen::Vector3d tmp = pt_data.t_bl.cross(rot_vel);
            // RCLCPP_INFO(rclcpp::get_logger("Estimator_dopp"), "txw: %f, %f, %f", tmp(0), tmp(1), tmp(2));

            pt_data.z_dopp = (R_IL * pt_data.pt_l).transpose() * (RT * v_itp - pt_data.t_bl.cross(rot_vel));
            pt_data.z_dopp /= pt_data.pt_l.norm();
            // RCLCPP_INFO(rclcpp::get_logger("Estimator_dopp"), "doppler est: %f", pt_data.z_dopp);
            // RCLCPP_INFO(rclcpp::get_logger("Estimator_dopp"), "doppler measure: %f", pt_data.pt.normal_z);

            pt_data.z_dopp += pt_data.pt.normal_z; // measurement : negative

            Eigen::Matrix<double, 3, 4> drot;
            Quater::drot(v_itp, q_itp, drot);

            Eigen::Matrix<double, 3, 3> skew_t_bl = Eigen::Matrix3d::Zero();
            skew_t_bl << 0, -pt_data.t_bl(2), pt_data.t_bl(1),
                        pt_data.t_bl(2), 0, -pt_data.t_bl(0),
                        -pt_data.t_bl(1), pt_data.t_bl(0), 0;

            int RCP_st_id = spl.numKnots() - 4;
            for (int i = 0; i < (int) J_vel.d_val_d_knot.size(); i++) {
                int j = (int) J_vel.start_idx + i - RCP_st_id;
                if (j >= 0) {
                    Hi.block(0, j*6, 1, 3) = (R_IL * pt_data.pt_l).transpose() * RT * J_vel.d_val_d_knot[i] / pt_data.pt_l.norm(); // trans
                    Hi.block(0, j*6 + 3, 1, 3) = (R_IL * pt_data.pt_l).transpose() * (drot * J_ortdel.d_val_d_knot[i] - skew_t_bl * J_gyro.d_val_d_knot[i]) / pt_data.pt_l.norm(); // rot
                }
            }  
            pt_data.H_dopp = Hi.template leftCols<24>();
        }
        
    }

    void updateState(const Eigen::Matrix<double, XSIZE, 1>& xupd)
    {
        Eigen::Matrix<double, CP_SIZE, 1> cp_win = xupd.segment(0, CP_SIZE);
        spl.updateRCPs(cp_win);
        if constexpr (XSIZE == 30) {
            ba = xupd.segment(BA_OFFSET, 3);
            for (int i = 0; i < 3; i++) {
                if (std::abs(ba(i)) > 0.2) {
                    ba(i) = 0.0; // reset bias if too large
                }
            }
            bg = xupd.segment(BG_OFFSET, 3);   
        }
    }        

    bool updateLiDAR(Eigen::aligned_deque<PointData>& pt_meas, int num_valid, const Eigen::Matrix<double, XSIZE, 1>& x_prop, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop, const double pt_thresh, const double cov_thresh)
    {
        
        size_t num_pt = pt_meas.size(); 
        
        valid_cnt = 0;
        plane_cnt = 0;
        
        for (size_t i = 0; i < num_pt; i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                valid_cnt++;
            }
            if (pt_data.is_plane) {
                plane_cnt++;
            }
        }

        #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
        for (size_t i = 0; i < num_pt; i++) {
            PointData& pt_data = pt_meas[i];
            prepLiDAR(pt_data);
            prepDoppler(pt_data); 
        }

        Eigen::Array<int,6,1> localizability_mat = Eigen::Array<int,6,1>::Zero();
        for (const auto &pt : pt_meas) {
            localizability_mat +=
                (pt.loc_contrib.array() > 0.5).template cast<int>();
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "localizability: %d, %d, %d, %d, %d, %d", 
        //     localizability_mat(0), localizability_mat(1), localizability_mat(2), 
        //     localizability_mat(3), localizability_mat(4), localizability_mat(5));
        
        // conduct the constraint
        std::vector<int> idxs;
        idxs.reserve(6);
        if (plane_cnt > valid_cnt * 0.5) {
            for (int i = 0; i < 6; ++i) {
                if (localizability_mat(i) < plane_cnt * 0.05) { // under-observable case (no contribution points)
                    idxs.push_back(i);
                }
            }
        }
        
        // Eigen::Matrix<double, Eigen::Dynamic, 24> constraint_mat(idxs.size(), 24);
        constraint_mat = Eigen::Matrix<double, Eigen::Dynamic, 24>::Zero(idxs.size(), 24);

        for (size_t k = 0; k < idxs.size(); ++k) {
            int i = idxs[k];
            constraint_mat(k, 18 + i) = 1.0;
        }

        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "constraint_mat:\n%s", 
        //     CommonUtils::eigenToString(constraint_mat).c_str());


        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "constraint_mat:\n%s", 
        //     CommonUtils::eigenToString(constraint_mat).c_str());

        // compute uncertainty cov
        std::vector<double> pt_uncs = std::vector<double>(pt_meas.size(), 0.0);
        std::vector<double> plane_covs = std::vector<double>(pt_meas.size(), 0.0);

        double max_cov = 0.0;
        double min_cov = 9999.0;
        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                pt_uncs[i] = pt_data.pt_unc;

                if (pt_data.pt_unc > max_cov) {
                    max_cov = pt_data.pt_unc;
                }
                if (pt_data.pt_unc < min_cov) {
                    min_cov = pt_data.pt_unc;
                }
            }
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "max pt_unc: %f, min pt_unc: %f", max_cov, min_cov);

        double cov_lo = min_cov_bound;
        double cov_hi = max_cov_bound;

        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                if (pt_data.pt_unc < cov_lo) {
                    pt_uncs[i] = min_cov_thresh;
                }
                else if (pt_data.pt_unc > cov_hi) {
                    pt_uncs[i] = max_cov_thresh;
                }
                else{
                    pt_uncs[i] = min_cov_thresh + (pt_data.pt_unc - cov_lo) / (cov_hi - cov_lo) * (max_cov_thresh - min_cov_thresh);
                }
            }
        }

        double max_plane_cov = 0.0;
        double min_plane_cov = 9999.0;
        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid && pt_data.is_plane) {
                plane_covs[i] = pt_data.plane_cov;
                if (pt_data.plane_cov > max_plane_cov) {
                    max_plane_cov = pt_data.plane_cov;
                }
                if (pt_data.plane_cov < min_plane_cov) {
                    min_plane_cov = pt_data.plane_cov;
                }
            }
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "max plane_cov: %f, min plane_cov: %f", max_plane_cov, min_plane_cov);

        double pcov_lo = min_plane_cov_bound;
        double pcov_hi = max_plane_cov_bound;

        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                if (pt_data.plane_cov < pcov_lo) {
                    plane_covs[i] = min_plane_cov_thresh;
                }
                else if (pt_data.plane_cov > pcov_hi) {
                    plane_covs[i] = max_plane_cov_thresh;
                }
                else{
                    plane_covs[i] = min_plane_cov_thresh + (pt_data.plane_cov - pcov_lo) / (pcov_hi - pcov_lo) * (max_plane_cov_thresh - min_plane_cov_thresh);
                }
            }
        }

        Eigen::Matrix<double, Eigen::Dynamic, XSIZE> H(2 * num_valid, XSIZE);
        Eigen::Matrix<double, Eigen::Dynamic, 1> innv(2 * num_valid, 1);
        Eigen::Matrix<double, Eigen::Dynamic, 1> mat_cov_inv(2 * num_valid, 1);
        H.setZero();    
        innv.setZero();
        mat_cov_inv.setZero();

        int idx_offset = 0;
        for(size_t i = 0; i < num_pt; i++) {
            const PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {

                Eigen::Matrix<double, 24, 24> cov = cov_rcp.template topLeftCorner<24, 24>();
                double lid_cov = pt_data.H*cov*pt_data.H.transpose() + pt_data.var_pt;
                const bool ok_pl = (std::abs(pt_data.zpl) < pt_thresh || lid_cov < pt_data.var_pt * cov_thresh);
                const bool ok_pt = (std::abs(pt_data.zp) < pt_thresh || lid_cov < pt_data.var_pt * cov_thresh);

                if(pt_data.is_plane){
                    if(ok_pl){
                        innv(idx_offset) = - pt_data.zpl * double(plane_cnt) / valid_cnt; // point to plane
                        H.block(idx_offset, 0, 1, 24) = pt_data.H_pl * double(plane_cnt) / valid_cnt;
                    }
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "planar point res: %f", pt_data.zpl);
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "planar point weight: %f", double(plane_cnt) / valid_cnt);
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "\n%s", CommonUtils::eigenToString(pt_data.H_pl).c_str());

                }
                else {
                    if(ok_pt){
                        innv(idx_offset) = - pt_data.zp * double(valid_cnt - plane_cnt) / valid_cnt; // point to point
                        H.block(idx_offset, 0, 1, 24) = pt_data.H * double(valid_cnt - plane_cnt) / valid_cnt;
                    }

                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "non planar point res: %f", pt_data.zp);
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "\n%s", CommonUtils::eigenToString(pt_data.H).c_str());
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "non planar point weight: %f", double(valid_cnt - plane_cnt) / valid_cnt);
                }

                if (pt_data.is_plane) {
                    mat_cov_inv(idx_offset) = 1/plane_covs[i]; // point to plane
                }
                else{
                    mat_cov_inv(idx_offset) = 1/pt_uncs[i]; // point to point
                }
                // mat_cov_inv(idx_offset) = 1/0.01;

                if (abs(pt_data.z_dopp) < 0.2) {
                    innv(idx_offset + 1) = - pt_data.z_dopp; // -, +
                    H.block(idx_offset + 1, 0, 1, 24) = pt_data.H_dopp;
                    if (pt_data.is_plane) {
                        innv(idx_offset + 1) *= double(plane_cnt) / valid_cnt; // weight the doppler residual
                        H.block(idx_offset + 1, 0, 1, 24) *= double(plane_cnt) / valid_cnt; // weight the doppler jacobian
                    } else {
                        innv(idx_offset + 1) *= double(valid_cnt - plane_cnt) / valid_cnt; // weight the doppler residual
                        H.block(idx_offset + 1, 0, 1, 24) *= double(valid_cnt - plane_cnt) / valid_cnt; // weight the doppler jacobian
                    }
                }
                // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "doppler res: %f", pt_data.z_dopp);
                // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "\n%s", CommonUtils::eigenToString(pt_data.H_dopp).c_str());

                mat_cov_inv(idx_offset + 1) = 1/pt_data.var_vel; // doppler

                // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "doppler res: %f", pt_data.z_dopp);
                // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "doppler cov: %f", pt_data.var_vel);

                // delete the jacobian in the constraint
                // for(size_t i = 0; i < idxs.size(); i++) {
                //     H(idx_offset, 18 + idxs[i]) = 0.0;
                //     H(idx_offset + 1, 18 + idxs[i]) = 0.0;
                // }

                idx_offset += 2;
            }
        }        
        update(innv, mat_cov_inv, H, x_prop, P_prop, &constraint_mat);
        return true;
    }           

    void updateLiDARInertial(Eigen::aligned_deque<PointData>& pt_meas, Eigen::aligned_deque<ImuData>& imu_meas, int num_valid, const Eigen::Matrix<double, XSIZE, 1>& x_prop, 
        const Eigen::Matrix<double, XSIZE, XSIZE>& P_prop, const double pt_thresh, const double cov_thresh, const Eigen::Vector3d& g, const Eigen::Vector3d& cov_acc, const Eigen::Vector3d& cov_gyro, const double& cov_grav = 0.1)
    {
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "***********************");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "* updateLiDARInertial *");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "***********************");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "Find correspondence, with %zu points", pt_meas.size());

        // auto fmt = Eigen::IOFormat(6, 0, ", ", "\n", "[", "]");

        Eigen::Matrix<double, 6, 1> cov_imu_inv =  Eigen::Matrix<double, 6, 1>(1/cov_acc[0], 1/cov_acc[1], 1/cov_acc[2], 1/cov_gyro[0], 1/cov_gyro[1], 1/cov_gyro[2]);
        
        // count the is valid 
        valid_cnt = 0;
        plane_cnt = 0;
        
        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                valid_cnt++;
            }
            if (pt_data.is_plane) {
                plane_cnt++;
            }
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "plane cnt: %d, non plane cnt: %d, total valid cnt: %d", 
        //     plane_cnt, valid_cnt - plane_cnt, valid_cnt);

        // TODO : parallelize this part
        #pragma omp parallel for num_threads(NUM_OF_THREAD) schedule(dynamic)
        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            prepLiDAR(pt_data);
            prepDoppler(pt_data); 
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "*******************");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "* prep LiDAR done *");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "*******************");

        // localizability?
        Eigen::Array<int,6,1> localizability_mat = Eigen::Array<int,6,1>::Zero();
        for (const auto &pt : pt_meas) {
            localizability_mat +=
                (pt.loc_contrib.array() > 0.5).template cast<int>();
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "localizability: %d, %d, %d, %d, %d, %d", 
        //     localizability_mat(0), localizability_mat(1), localizability_mat(2), 
        //     localizability_mat(3), localizability_mat(4), localizability_mat(5));
        
        // conduct the constraint
        std::vector<int> idxs;
        idxs.reserve(6);
        if (plane_cnt > valid_cnt * 0.5) {
            for (int i = 0; i < 6; ++i) {
                if (localizability_mat(i) < plane_cnt * 0.05) { // under-observable case (no contribution points)
                    idxs.push_back(i);
                }
            }
        }
        
        // Eigen::Matrix<double, Eigen::Dynamic, 24> constraint_mat(idxs.size(), 24);
        constraint_mat = Eigen::Matrix<double, Eigen::Dynamic, 24>::Zero(idxs.size(), 24);

        for (size_t k = 0; k < idxs.size(); ++k) {
            int i = idxs[k];
            constraint_mat(k, 18 + i) = 1.0;
        }

        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "constraint_mat:\n%s", 
        //     CommonUtils::eigenToString(constraint_mat).c_str());

        

        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "constraint_mat:\n%s", 
        //     CommonUtils::eigenToString(constraint_mat).c_str());

        // compute uncertainty cov
        std::vector<double> pt_uncs = std::vector<double>(pt_meas.size(), 0.0);
        std::vector<double> plane_covs = std::vector<double>(pt_meas.size(), 0.0);

        double max_cov = 0.0;
        double min_cov = 9999.0;
        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                pt_uncs[i] = pt_data.pt_unc;

                if (pt_data.pt_unc > max_cov) {
                    max_cov = pt_data.pt_unc;
                }
                if (pt_data.pt_unc < min_cov) {
                    min_cov = pt_data.pt_unc;
                }
            }
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "max pt_unc: %f, min pt_unc: %f", max_cov, min_cov);

        double cov_lo = min_cov_bound;
        double cov_hi = max_cov_bound;

        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                if (pt_data.pt_unc < cov_lo) {
                    pt_uncs[i] = min_cov_thresh;
                }
                else if (pt_data.pt_unc > cov_hi) {
                    pt_uncs[i] = max_cov_thresh;
                }
                else{
                    pt_uncs[i] = min_cov_thresh + (pt_data.pt_unc - cov_lo) / (cov_hi - cov_lo) * (max_cov_thresh - min_cov_thresh);
                }
            }
        }

        double max_plane_cov = 0.0;
        double min_plane_cov = 9999.0;
        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid && pt_data.is_plane) {
                plane_covs[i] = pt_data.plane_cov;
                if (pt_data.plane_cov > max_plane_cov) {
                    max_plane_cov = pt_data.plane_cov;
                }
                if (pt_data.plane_cov < min_plane_cov) {
                    min_plane_cov = pt_data.plane_cov;
                }
            }
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "max plane_cov: %f, min plane_cov: %f", max_plane_cov, min_plane_cov);

        double pcov_lo = min_plane_cov_bound;
        double pcov_hi = max_plane_cov_bound;

        for (size_t i = 0; i < pt_meas.size(); i++) {
            PointData& pt_data = pt_meas[i];
            if (pt_data.if_valid) {
                if (pt_data.plane_cov < pcov_lo) {
                    plane_covs[i] = min_plane_cov_thresh;
                }
                else if (pt_data.plane_cov > pcov_hi) {
                    plane_covs[i] = max_plane_cov_thresh;
                }
                else{
                    plane_covs[i] = min_plane_cov_thresh + (pt_data.plane_cov - pcov_lo) / (pcov_hi - pcov_lo) * (max_plane_cov_thresh - min_plane_cov_thresh);
                }
            }
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "uncertainty cov computed");

        // TODO : parallelize this part
        #pragma omp parallel for num_threads(NUM_OF_THREAD) 
        for (size_t i = 0; i < imu_meas.size(); i++) {
            prepIMU(imu_meas[i], g);
        }
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "*****************");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "* prep IMU done *");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "*****************");

        int dim_meas = 7 * imu_meas.size() + 2 * num_valid;
        Eigen::Matrix<double, Eigen::Dynamic, XSIZE> H(dim_meas, XSIZE);
        Eigen::Matrix<double, Eigen::Dynamic, 1> innv(dim_meas, 1);
        Eigen::Matrix<double, Eigen::Dynamic, 1> mat_cov_inv(dim_meas, 1);
        H.setZero();    
        innv.setZero();
        mat_cov_inv.setZero();
        int idx_offset = 0;
        size_t id_imu = 0;
        size_t id_pt = 0;

        for (size_t j = 0; j < imu_meas.size() + pt_meas.size(); j++) {
            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "*****************");
            if ((id_pt < pt_meas.size() && id_imu < imu_meas.size() && pt_meas[id_pt].time_ns < imu_meas[id_imu].time_ns) ||
                (id_pt < pt_meas.size() && id_imu >= imu_meas.size())) {
                    PointData& pt_data = pt_meas[id_pt];
                    if (pt_data.if_valid) {
                        Eigen::Matrix<double, 24, 24> cov = cov_rcp.template topLeftCorner<24, 24>();
                        double lid_cov = pt_data.H*cov*pt_data.H.transpose() + pt_data.var_pt;
                        const bool ok_pl = (std::abs(pt_data.zpl) < pt_thresh || lid_cov < pt_data.var_pt * cov_thresh);
                        const bool ok_pt = (std::abs(pt_data.zp) < pt_thresh || lid_cov < pt_data.var_pt * cov_thresh);

                        double vel_cov = pt_data.H_dopp*cov*pt_data.H_dopp.transpose() + pt_data.var_pt;

                        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "lid cov: %f", lid_cov);
                        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "cov_thresh: %f", pt_data.var_pt*cov_thresh);
                        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "vel cov: %f", vel_cov);

                        // if (abs(pt_data.zp) < pt_thresh || lid_cov < pt_data.var_pt * cov_thresh) {

                        if(pt_data.is_plane){
                            if(ok_pl){
                                innv(idx_offset) = - pt_data.zpl * double(plane_cnt) / valid_cnt; // point to plane
                                H.block(idx_offset, 0, 1, 24) = pt_data.H_pl * double(plane_cnt) / valid_cnt;
                            }
                            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "planar point res: %f", pt_data.zpl);
                            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "planar point weight: %f", double(plane_cnt) / valid_cnt);
                            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "\n%s", CommonUtils::eigenToString(pt_data.H_pl).c_str());

                        }
                        else {
                            if(ok_pt){
                                innv(idx_offset) = - pt_data.zp * double(valid_cnt - plane_cnt) / valid_cnt; // point to point
                                H.block(idx_offset, 0, 1, 24) = pt_data.H * double(valid_cnt - plane_cnt) / valid_cnt;
                            }

                            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "non planar point res: %f", pt_data.zp);
                            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "\n%s", CommonUtils::eigenToString(pt_data.H).c_str());
                            // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "non planar point weight: %f", double(valid_cnt - plane_cnt) / valid_cnt);
                        }

                        if (pt_data.is_plane) {
                            mat_cov_inv(idx_offset) = 1/plane_covs[id_pt]; // point to plane
                        }
                        else{
                            mat_cov_inv(idx_offset) = 1/pt_uncs[id_pt]; // point to point
                        }
                        // mat_cov_inv(idx_offset) = 1/0.01;

                        if (abs(pt_data.z_dopp) < 0.2) {
                            innv(idx_offset + 1) = - pt_data.z_dopp; // -, +
                            H.block(idx_offset + 1, 0, 1, 24) = pt_data.H_dopp;
                            if (pt_data.is_plane) {
                                innv(idx_offset + 1) *= double(plane_cnt) / valid_cnt; // weight the doppler residual
                                H.block(idx_offset + 1, 0, 1, 24) *= double(plane_cnt) / valid_cnt; // weight the doppler jacobian
                            } else {
                                innv(idx_offset + 1) *= double(valid_cnt - plane_cnt) / valid_cnt; // weight the doppler residual
                                H.block(idx_offset + 1, 0, 1, 24) *= double(valid_cnt - plane_cnt) / valid_cnt; // weight the doppler jacobian
                            }
                        }
                        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "doppler res: %f", pt_data.z_dopp);
                        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "\n%s", CommonUtils::eigenToString(pt_data.H_dopp).c_str());

                        mat_cov_inv(idx_offset + 1) = 1/pt_data.var_vel; // doppler

                        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "doppler res: %f", pt_data.z_dopp);
                        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "doppler cov: %f", pt_data.var_vel);

                        // delete the jacobian in the constraint
                        // for(size_t i = 0; i < idxs.size(); i++) {
                        //     H(idx_offset, 18 + idxs[i]) = 0.0;
                        //     H(idx_offset + 1, 18 + idxs[i]) = 0.0;
                        // }

                        idx_offset += 2;
                    }
                    id_pt++;
            } else if ((id_pt < pt_meas.size() && id_imu < imu_meas.size() && pt_meas[id_pt].time_ns >= imu_meas[id_imu].time_ns) ||
                        (id_pt >= pt_meas.size() && id_imu < imu_meas.size())) {
                    const ImuData& imu_data = imu_meas[id_imu];
                    Eigen::Matrix<double, 6, 1> imu_itp = imu_data.imu_itp;
                    Eigen::Matrix<double, 6, XSIZE> Hi = Eigen::Matrix<double, 6, XSIZE>::Zero();
                    Hi.template leftCols<24>() = imu_data.H;
                    Hi.block(0, BA_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();
                    Hi.block(3, BG_OFFSET, 3, 3) = Eigen::Matrix3d::Identity();   

                    Eigen::Matrix<double, 6, 1> imu;
                    imu.head<3>() = imu_data.accel;
                    imu.tail<3>() = imu_data.gyro;
                    for (int i = 0; i < 3; i++) {
                        if (abs(imu(i) - imu_itp(i)) > 5.0) { // acc
                            imu(i) = 0;
                            imu_itp(i) = 0;
                            Hi.row(i).setZero();
                        } 
                        if (abs(imu(i+3) - imu_itp(i+3)) > 5.0) { // omg
                            imu(i+3) = 0;
                            imu_itp(i+3) = 0;
                            Hi.row(i+3).setZero();
                        }                     
                    }
                    innv.segment<6>(idx_offset) = imu - imu_itp;
                    H.block(idx_offset, 0, 6, XSIZE) = Hi;
                    mat_cov_inv.segment<6>(idx_offset) = cov_imu_inv;

                    if(imu_data.z_grav < 0.1 && imu_data.z_grav > 0) {
                        innv(idx_offset + 6) = - imu_data.z_grav; // gravity
                        H.block(idx_offset + 6, 0, 1, XSIZE) = imu_data.H_grav;
                    }
                    mat_cov_inv(idx_offset + 6) = 1.0 / cov_grav;

                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "imu residual: %f, %f, %f, %f, %f, %f", 
                    //     imu(0) - imu_itp(0), imu(1) - imu_itp(1), imu(2) - imu_itp(2), imu(3) - imu_itp(3), imu(4) - imu_itp(4), imu(5) - imu_itp(5));
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "\n%s", CommonUtils::eigenToString(imu_data.H).c_str());
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "imu cov: %f, %f, %f, %f, %f, %f",
                    //     1 / mat_cov_inv(idx_offset), 1 / mat_cov_inv(idx_offset + 1), 1 / mat_cov_inv(idx_offset + 2), 
                    //     1 / mat_cov_inv(idx_offset + 3), 1 / mat_cov_inv(idx_offset + 4), 1 / mat_cov_inv(idx_offset + 5));

                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "gravity residual: %f", imu_data.z_grav);
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "\n%s", CommonUtils::eigenToString(imu_data.H_grav).c_str());
                    // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "gravity cov: %f", 1 / mat_cov_inv(idx_offset + 6));

                    innv.segment<3>(idx_offset) = Eigen::Vector3d::Zero();
                    H.block(idx_offset, 0, 3, XSIZE) = Eigen::Matrix<double, 3, XSIZE>::Zero();

                    idx_offset += 7; // 6 imu + 1 gravity
                    id_imu++;
            }
        }

        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "*************************");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "* ALL residual computed *");
        // RCLCPP_INFO(rclcpp::get_logger("Estimator"), "*************************");
        update(innv, mat_cov_inv, H, x_prop, P_prop, &constraint_mat);        
    }

    template <int RSIZE>
    void update(const Eigen::Matrix<double, RSIZE, 1>& innov, const Eigen::Matrix<double, RSIZE, 1>& R_inv, const Eigen::Matrix<double, RSIZE, XSIZE>& H, 
        const Eigen::Matrix<double, XSIZE, 1>& x_prop, const Eigen::Matrix<double, XSIZE, XSIZE>& cov_prop, const Eigen::MatrixXd* constraint_mat=nullptr)
    {
        int num_pts = innov.rows();
        Eigen::Matrix<double, XSIZE, 1> RCPs_post;
        Eigen::MatrixXd I_X = Eigen::MatrixXd::Identity(XSIZE, XSIZE); 
        if (num_pts > XSIZE) {
            Eigen::Matrix<double, XSIZE, XSIZE> cov_rcp_inv = cov_prop.llt().solve(I_X);
            Eigen::Matrix<double, XSIZE, RSIZE> HT_R_inv;
            HT_R_inv.noalias() = (H.transpose().array().rowwise() * R_inv.transpose().array()).matrix();
            Eigen::Matrix<double, XSIZE, XSIZE> HT_R_inv_H;
            HT_R_inv_H.noalias() = HT_R_inv * H;

            Eigen::Matrix<double, XSIZE, XSIZE> S = HT_R_inv_H;
            S.noalias() += cov_rcp_inv;
            Eigen::Matrix<double, XSIZE, XSIZE> S_inv = S.llt().solve(I_X);
            Eigen::Matrix<double, XSIZE, RSIZE> K;
            K.noalias() = S_inv * HT_R_inv;

            KH.noalias() = S_inv * HT_R_inv_H;     
            Eigen::Matrix<double, XSIZE, 1> delta_cur = (getState() - x_prop);
            Eigen::Matrix<double, XSIZE, 1> deltax = KH * delta_cur + K * innov - delta_cur;

            if (constraint_mat && constraint_mat->rows() > 0) {
                const auto& C = *constraint_mat;
                Eigen::MatrixXd B = C.rightCols(6);   // k×6

                // lock the update for under-estimated directio for last cp // last two?
                for (int i = 3; i < 4; i++) {    
                    Eigen::VectorXd dx_partial = deltax.template segment<6>(i * 6);  // 6×1
    
                    // Compute violation and solve for λ
                    Eigen::VectorXd v      = B * dx_partial;                     // k×1
                    Eigen::VectorXd lambda = (B * B.transpose()).ldlt().solve(v); // k×1
    
                    // Project out the violation
                    dx_partial.noalias() -= B.transpose() * lambda;
    
                    // Write back into deltax[18…23]
                    deltax.template segment<6>(i * 6) = dx_partial;
                }
            }

            RCPs_post.noalias() = getState() + deltax;
        } else {
            Eigen::Matrix<double, RSIZE, RSIZE> R = R_inv.cwiseInverse().asDiagonal();
            Eigen::Matrix<double, RSIZE, RSIZE> S;
            S.noalias() = H * cov_prop * H.transpose() + R;
            Eigen::Matrix<double, XSIZE, RSIZE> K;
            K.noalias() = cov_prop * H.transpose() * S.inverse();
            KH.noalias() = K * H;
            Eigen::Matrix<double, XSIZE, 1> delta_cur = (getState() - x_prop);
            Eigen::Matrix<double, XSIZE, 1> deltax = KH * delta_cur + K * innov - delta_cur;

            // if (constraint_mat && constraint_mat->rows() > 0) {
            //     const auto& C = *constraint_mat;
            //     Eigen::MatrixXd B = C.rightCols(6);   // k×6

            //     // 2) Extract Δx[18…23]
            //     Eigen::VectorXd dx_tail = deltax.template segment<6>(18);  // 6×1

            //     // 3) Compute violation and solve for λ
            //     Eigen::VectorXd v      = B * dx_tail;                     // k×1
            //     Eigen::VectorXd lambda = (B * B.transpose()).ldlt().solve(v); // k×1

            //     // 4) Project out the violation
            //     dx_tail.noalias() -= B.transpose() * lambda;

            //     // 5) Write back into deltax[18…23]
            //     deltax.template segment<6>(18) = dx_tail;
            // }

            RCPs_post.noalias() = getState() + deltax;

        }
        updateState(RCPs_post);     
    }
    
    
};
