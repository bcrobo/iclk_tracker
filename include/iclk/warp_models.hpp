#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>

class Affine2D : public Eigen::Transform<double, 2, Eigen::Affine> {
public:
  enum { DoF = 6 };
  using Base = Eigen::Transform<double, 2, Eigen::Affine>;
  using Scalar = Base::Scalar;
  using MatrixType = Base::MatrixType;
  using Jacobian = Eigen::Matrix<Scalar, 2, DoF>;
  using ParameterVector = Eigen::Matrix<Scalar, DoF, 1>;
  using Hessian = Eigen::Matrix<Scalar, DoF, DoF>;

  template <typename Derived> Affine2D(const Derived &b) : Base(b) {}

  static Affine2D fromParameters(const ParameterVector &p) {
    auto transform = Base{};
    // clang-format off
    transform.matrix() << (Base::MatrixType() <<
		    1 + p(0), 	p(2), 		p(4),
		    p(1),	1 + p(3), 	p(5),
		    0, 		0, 		1).finished();
    // clang-format on
    return transform;
  }

  static Jacobian jacobian(const Affine2D &,
                           const Eigen::Matrix<Scalar, 2, 1> &p) {
    // clang-format off
    return (Jacobian() << 
		    p(0), 0,    p(1), 0,    1, 0,
		    0,    p(0), 0,    p(1), 0, 1).finished();
    // clang-format on
  }

  static ParameterVector parameterInverse(const ParameterVector &p) {
    const auto &p1 = p(0);
    const auto &p2 = p(1);
    const auto &p3 = p(2);
    const auto &p4 = p(3);
    const auto &p5 = p(4);
    const auto &p6 = p(5);

    const auto det = 1 / ((1 + p1) * (1 + p4) - p2 * p3);
    if (std::abs(det) < 1e-10) {
      throw std::runtime_error{
          "Inverse warp parameters is degenerate (not invertible)"};
    }

    // clang-format off
    const auto dp = (ParameterVector() <<
		    -p1 - p1 * p4 + p2 * p3,
		    -p2,
		    -p3,
                    -p4 - p1 * p4 + p2 * p3,
		    -p5 - p4 * p5 + p3 * p6,
                     -p6 - p1 * p6 + p2 * p5).finished();
    // clang-format on
    return det * dp;
  }

  Affine2D inverse() const { return matrix().inverse(); }
  void translateWarp(Scalar x, Scalar y) {
    matrix()(0, 2) += x;
    matrix()(1, 2) += y;
  }
  void scaleWarp(Scalar scale_factor) { translation() *= scale_factor; }

};

class Translation2D : public Eigen::Transform<double, 2, Eigen::Affine> {
public:
  enum { DoF = 2 };
  using Base = Eigen::Transform<double, 2, Eigen::Affine>;
  using Scalar = Base::Scalar;
  using MatrixType = Base::MatrixType;
  using Jacobian = Eigen::Matrix<Scalar, 2, DoF>;
  using ParameterVector = Eigen::Matrix<Scalar, DoF, 1>;
  using Hessian = Eigen::Matrix<Scalar, DoF, DoF>;

  Translation2D() : Base() {}

  template <typename Derived> Translation2D(const Derived &b) : Base(b) {}

  static Translation2D fromParameters(const ParameterVector &p) {
    auto transform = Base{};
    transform.setIdentity();
    transform.matrix()(0, 2) = p(0);
    transform.matrix()(1, 2) = p(1);
    return transform;
  }

  static Jacobian jacobian(const Translation2D &,
                           const Eigen::Matrix<Scalar, 2, 1> &) {
    return Jacobian::Identity();
  }

  static ParameterVector parameterInverse(const ParameterVector &p) {
    return (ParameterVector() << -p(0), -p(1)).finished();
  }

  void scaleWarp(Scalar scale_factor) { translation() *= scale_factor; }

  void translateWarp(Scalar x, Scalar y) {
    matrix()(0, 2) += x;
    matrix()(1, 2) += y;
  }

  Translation2D inverse() const {
    auto T = *this;
    T.translation() = -T.translation();
    return T;
  }
};

class Sim2 : public Eigen::Transform<double, 2, Eigen::Affine> {
public:
  enum { DoF = 4 };
  using Base = Eigen::Transform<double, 2, Eigen::Affine>;
  using Scalar = Base::Scalar;
  using MatrixType = Base::MatrixType;
  using RotationMatrix = Eigen::Matrix<Scalar, 2, 2>;
  using Translation = Eigen::Matrix<Scalar, 2, 1>;
  using Jacobian = Eigen::Matrix<Scalar, 2, DoF>;
  using ParameterVector = Eigen::Matrix<Scalar, DoF, 1>;
  using Hessian = Eigen::Matrix<Scalar, DoF, DoF>;

  template <typename Derived> Sim2(const Derived &b) : Base(b) {}

  static Sim2 fromParameters(const ParameterVector &p) {
    auto transform = Base{};
    transform.setIdentity();
    // Parameter vector [upsilon_x, upsilon_y, theta, sigma]
    const auto upsilon_x = p(0);
    const auto upsilon_y = p(1);
    const auto upsilon = (Translation() << upsilon_x, upsilon_y).finished();
    const auto theta = p(2);
    const auto ctheta = std::cos(theta);
    const auto stheta = std::sin(theta);
    // scale = exp(sigma)
    const auto s = std::exp(p(3));

    auto sR = (RotationMatrix() << ctheta, -stheta, stheta, ctheta).finished();
    sR *= s;

    const auto omega = (RotationMatrix() << 0, -theta, theta, 0).finished();
    // Picked from sophus c++
    const auto W = Sim2::calcW<Scalar, 2>(omega, theta, s);
    transform.matrix().block<2, 2>(0, 0) = sR;
    transform.matrix().block<2, 1>(0, 2) = W * upsilon;
    return transform;
  }

  static Jacobian jacobian(const Sim2&,
                           const Eigen::Matrix<Scalar, 2, 1> &uv) {
    // clang-format off
    static auto J_exp_x_at_0 = (Eigen::Matrix<Scalar, 4, 4>() <<
		    0, 0, 0, 1,
		    0, 0, 1, 0,
		    1, 0, 0, 0,
		    0, 1, 0, 0).finished();

    const auto J_warp_at_0 =
        (Jacobian() <<
	 1, 0, -uv(1), uv(0),
	 0, 1, uv(0),  uv(1)).finished();
    // clang-format on
    return J_warp_at_0;
  }

  void scaleWarp(Scalar scale_factor) { translation() *= scale_factor; }
  void translateWarp(Scalar x, Scalar y) {
    matrix()(0, 2) += x;
    matrix()(1, 2) += y;
  }

  Sim2 inverse() const {
    auto T = Base{};
    T.setIdentity();
    // Inverse RxSO2 (s * R) part of Sim2
    auto exp_sigma_times_ctheta = matrix()(0, 0);
    auto exp_sigma_times_stheta = matrix()(1, 0);
    const auto squared_scale = exp_sigma_times_ctheta * exp_sigma_times_ctheta +
                               exp_sigma_times_stheta * exp_sigma_times_stheta;
    exp_sigma_times_ctheta /= squared_scale;
    exp_sigma_times_stheta /= squared_scale;
    const auto new_exp_sigma_times_ctheta = exp_sigma_times_ctheta;
    const auto new_exp_sigma_times_stheta = -exp_sigma_times_stheta;
    // \TODO check scale variability against epsilon
    // Conjugate of complex is inverse rotation

    // clang-format off
    const auto inv_sR = (RotationMatrix() <<
		    new_exp_sigma_times_ctheta, -new_exp_sigma_times_stheta,
                    new_exp_sigma_times_stheta, new_exp_sigma_times_ctheta).finished();
    // clang-format on
    T.matrix().block<2, 2>(0, 0) = inv_sR;
    T.matrix().block<2, 1>(0, 2) = -inv_sR * translation();
    return T;
  }

  template <class Scalar, int N>
  static Eigen::Matrix<Scalar, N, N>
  calcW(Eigen::Matrix<Scalar, N, N> const &Omega, const Scalar theta,
        const Scalar sigma) {
    static Eigen::Matrix<Scalar, N, N> const I =
        Eigen::Matrix<Scalar, N, N>::Identity();
    static Scalar const one(1);
    static Scalar const half(0.5);
    constexpr Scalar epsilon{1e-10};
    Eigen::Matrix<Scalar, N, N> const Omega2 = Omega * Omega;
    Scalar const scale = std::exp(sigma);
    Scalar A, B, C;
    if (std::abs(sigma) < epsilon) {
      C = one;
      if (std::abs(theta) < epsilon) {
        A = half;
        B = Scalar(1. / 6.);
      } else {
        Scalar theta_sq = theta * theta;
        A = (one - cos(theta)) / theta_sq;
        B = (theta - sin(theta)) / (theta_sq * theta);
      }
    } else {
      C = (scale - one) / sigma;
      if (std::abs(theta) < epsilon) {
        Scalar sigma_sq = sigma * sigma;
        A = ((sigma - one) * scale + one) / sigma_sq;
        B = (scale * half * sigma_sq + scale - one - sigma * scale) /
            (sigma_sq * sigma);
      } else {
        Scalar theta_sq = theta * theta;
        Scalar a = scale * std::sin(theta);
        Scalar b = scale * std::cos(theta);
        Scalar c = theta_sq + sigma * sigma;
        A = (a * sigma + (one - b) * theta) / (theta * c);
        B = (C - ((b - one) * sigma + a * theta) / (c)) * one / (theta_sq);
      }
    }
    return A * Omega + B * Omega2 + C * I;
  }
};

class SL3 : public Eigen::Transform<double, 2, Eigen::Projective> {
public:
  enum { DoF = 8 };
  using Base = Eigen::Transform<double, 2, Eigen::Projective>;
  using Scalar = Base::Scalar;
  using MatrixType = Base::MatrixType;
  using Jacobian = Eigen::Matrix<Scalar, 2, DoF>;
  using JacobianExp = Eigen::Matrix<Scalar, 9, 8>;
  using JacobianH = Eigen::Matrix<Scalar, 2, 9>;
  using ParameterVector = Eigen::Matrix<Scalar, DoF, 1>;
  using Hessian = Eigen::Matrix<Scalar, DoF, DoF>;
  using Generator = Eigen::Matrix<Scalar, 3, 3>;

  SL3() : Base() {}

  template <typename Derived> SL3(const Derived &b) : Base(b) {}

  static Eigen::Transform<Scalar, Dim, Mode, Options>
  fromParameters(const ParameterVector &p) {
    auto transform = Base{};
    transform.matrix() = SL3::exp(p);
    return transform;
  }

  static Jacobian jacobian(const SL3 &, const Eigen::Matrix<Scalar, 2, 1> &uv) {
    return SL3::jacH(uv) * SL3::m_Jg;
  }

  void scaleWarp(Scalar scale_factor) {
    // https://stackoverflow.com/questions/21019338/how-to-change-the-homography-with-the-scale-of-the-image
    matrix()(0, 2) *= scale_factor;
    matrix()(1, 2) *= scale_factor;
    matrix()(2, 0) /= scale_factor;
    matrix()(2, 1) /= scale_factor;
  }

  void translateWarp(Scalar x, Scalar y) {
    const auto T = (MatrixType() << 1, 0, x, 0, 1, y, 0, 0, 1).finished();
    matrix() = T * matrix();
  }

  SL3 inverse() const { return matrix().inverse(); }

private:
  /** @brief SL3 lie algebra generators */
  static const std::array<Generator, 8> m_generators;
  /** @brief Jacobian wrt the parameter for small sl3 increments */
  static const JacobianExp m_Jg;

  /** @brief Pass a parameter vector to the SL3 lie algebra */
  static MatrixType hat(const ParameterVector &p) {
    MatrixType A = MatrixType::Zero();
    for (auto i = 0u; i < std::size(m_generators); ++i) {
      A += p(i) * m_generators.at(i);
    }
    return A;
  }

  /** @brief Exp map for SL3 */
  static MatrixType exp(const ParameterVector &p, std::size_t order = 9) {
    // Lie algebra
    const auto A = SL3::hat(p);

    // Exp map expansion at provided order
    MatrixType G = MatrixType::Zero();
    MatrixType A_power_i = MatrixType::Identity();
    auto factor_i = Scalar{1};

    for (auto i = 0u; i < order; ++i) {
      G += (Scalar{1} / factor_i) * A_power_i;
      A_power_i *= A;
      factor_i *= (i + 1);
    }
    return G;
  }

  /** @brief Calculates the jacobian of x'=H(p)x at parameters p = 0 (i.e. H=I)
   */
  static JacobianH jacH(const Eigen::Matrix<Scalar, 2, 1> &uv) {
    JacobianH J = JacobianH::Zero();
    const auto uv1_t =
        (Eigen::Matrix<Scalar, 1, 3>() << uv(0), uv(1), 1).finished();
    J.block<1, 3>(0, 0) = uv1_t;
    J.block<1, 3>(1, 3) = uv1_t;
    J.block<1, 3>(0, 6) = -uv(0) * uv1_t;
    J.block<1, 3>(1, 6) = -uv(1) * uv1_t;
    return J;
  }
};

const std::array<SL3::Generator, 8> SL3::m_generators = {
    (Generator() << 0, 0, 1, 0, 0, 0, 0, 0, 0).finished(),
    (Generator() << 0, 0, 0, 0, 0, 1, 0, 0, 0).finished(),
    (Generator() << 0, 1, 0, 0, 0, 0, 0, 0, 0).finished(),
    (Generator() << 0, 0, 0, 1, 0, 0, 0, 0, 0).finished(),
    (Generator() << 1, 0, 0, 0, -1, 0, 0, 0, 0).finished(),
    (Generator() << 0, 0, 0, 0, -1, 0, 0, 0, 1).finished(),
    (Generator() << 0, 0, 0, 0, 0, 0, 1, 0, 0).finished(),
    (Generator() << 0, 0, 0, 0, 0, 0, 0, 1, 0).finished()};

// clang-format off
const SL3::JacobianExp SL3::m_Jg =
    (SL3::JacobianExp() <<
     	0, 0, 0, 0, 1, 0, 0, 0,
	0, 0, 1, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 1, 0, 0, 0, 0,
	0, 0, 0, 0, -1, -1, 0, 0,
	0, 1, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 0, 1, 0, 0).finished();
// clang-format on
