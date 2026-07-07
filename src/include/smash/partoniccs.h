/*
 *    Copyright (C) 2026
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 */

#ifndef SRC_INCLUDE_SMASH_PARTONICCS_H_
#define SRC_INCLUDE_SMASH_PARTONICCS_H_

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace smash {

/// Minimum electron energy threshold
const double pemin = 1.0e-5;
/// Number of cross-section grid points
extern int npxs;
/// Number of t-hat grid points for coarse-grid integration
extern int nth;
/// Integration dimension (5 * nth) for fine-grid t-hat integration
extern int intdim;

/**
 * \brief Struct to hold t-hat kinematic limits
 * 
 * Stores the minimum and maximum values of the Mandelstam variable t-hat,
 * along with a validity flag indicating whether kinematic phase space is open.
 */
struct TLimits {
  /// Minimum t-hat value [GeV²]
  double tmin;
  /// Maximum t-hat value [GeV²]
  double tmax;
  /// Boolean flag: true if limits are valid (phase space is open)
  bool ok;
};

/**
 * \brief Global variables storage for cross-section computations
 * 
 * Manages static arrays for process information, cumulative distributions,
 * and cross-section data used in sampling calculations.
 */
struct GlobalVar {
  /// Subprocess identifier array
  static std::vector<std::array<int, 3>> isig;
  /// Coarse-grid T integral array
  static std::vector<double> FTH;
  /// Fine-grid T integral array
  static std::vector<double> FTH2;
  /// Coarse-grid xi variable array
  static std::vector<double> VTH;
  /// Fine-grid xi variable array
  static std::vector<double> VTH2;
  /// Coarse-grid t-hat array
  static std::vector<double> TTH;
  /// Fine-grid t-hat array
  static std::vector<double> TTH2;
  /// Cross-section array indexed by subprocess
  static std::vector<double> xsec;
  /// Channel weights for subprocesses
  static std::vector<double> sigh;
};

/**
 * \brief Partonic cross-section calculator
 * 
 * Computes differential and integrated cross-sections for 2 → 2 partonic
 * scattering processes. Implements Mandelstam variable transformations and
 * provides sampling functionality for kinematic generation.
 */
class PartonicCS {

 public:
  /// Maximum iterations for root-finding algorithms
  static constexpr int maxit = 150;
  /// Relative accuracy tolerance for convergence
  static constexpr double xacc = 1e-5;
  /// Absolute accuracy tolerance for convergence
  static constexpr double yacc = 1e-20;
  /// Conversion factor (π) for cross-section calculations
  static constexpr double comfac = 3.141592653589793;
  /// Switch for t-channel color factors (mstp(34) in PYTHIA)
  static constexpr int mstp34 = 1;

  /**
   * \brief Compute differential cross-section for a subprocess
   * 
   * Evaluates \f$ d\sigma/dt \f$ for a given 2 → 2 partonic process
   * at specified Mandelstam variables.
   * 
   * \param[in] isub Subprocess identifier (1-10)
   * \param[in] kf1 PDG code of incoming parton 1
   * \param[in] kf2 PDG code of incoming parton 2
   * \param[in] shat Mandelstam variable \f$ \hat{s} \f$ [GeV²]
   * \param[in] that Mandelstam variable \f$ \hat{t} \f$ [GeV²]
   * \param[in] Q2 Virtuality scale \f$ Q^2 \f$ [GeV²]
   * \param[out] nchn Number of contributing channels
   * \param[out] dsigdt Differential cross-section [GeV⁻²]
   */
  void clox(int isub, int kf1, int kf2, double shat, double that, double Q2,
            int& nchn, double& dsigdt);

  /**
   * \brief Transform t-hat to xi variable
   * 
   * Computes the Jacobian \f$ dt/d\xi \f$ and xi variable for a subprocess,
   * enabling efficient integration over kinematic variables.
   * 
   * \param[in] isub Subprocess identifier
   * \param[in] kf1 PDG code of incoming parton 1
   * \param[in] kf2 PDG code of incoming parton 2
   * \param[in] shat Mandelstam variable \f$ \hat{s} \f$ [GeV²]
   * \param[in] that Mandelstam variable \f$ \hat{t} \f$ [GeV²]
   * \param[out] dtdxi Jacobian \f$ dt/d\xi \f$
   * \param[out] xi Transformed integration variable
   */
  void bmsxi(int isub, int kf1, int kf2, double shat, double that,
             double& dtdxi, double& xi);

  /**
   * \brief Inverse transformation from xi to t-hat
   * 
   * Uses improved Newton method with bisection to solve for t-hat
   * corresponding to a target value in xi.
   * 
   * \param[in] isub Subprocess identifier
   * \param[in] kf1 PDG code of incoming parton 1
   * \param[in] kf2 PDG code of incoming parton 2
   * \param[in] shat Mandelstam variable \f$ \hat{s} \f$ [GeV²]
   * \param[in] xi_target Target value of xi variable
   * \param[in] tmin Lower bound of t-hat range [GeV²]
   * \param[in] tmax Upper bound of t-hat range [GeV²]
   * \return t-hat value corresponding to xi_target
   * \throw std::runtime_error if iteration limit exceeded
   */
  double bmstofxi(int isub, int kf1, int kf2, double shat,
                  double xi_target, double tmin, double tmax);

  /**
   * \brief Compute running strong coupling constant
   * 
   * Evaluates \f$ \alpha_s(Q^2) \f$ at leading order using the
   * QCD running coupling formula.
   * 
   * \param[in] Q2 Renormalization scale \f$ Q^2 \f$ [GeV²]
   * \param[in] LambdaQCD QCD scale parameter [GeV]
   * \param[in] nf Number of active quark flavors
   * \return Strong coupling \f$ \alpha_s(Q^2) \f$
   * \throw std::invalid_argument if Q² ≤ 0
   * \throw std::domain_error if \f$ Q^2 \leq \Lambda_{\text{QCD}}^2 \f$
   */
  double alphas(double Q2, double LambdaQCD = 0.2, int nf = 3);

  /**
   * \brief Sample cross-section and generate kinematics
   * 
   * Computes total cross-section for a collision and samples kinematics
   * of outgoing partons according to the differential distribution.
   * 
   * \param[in] kf1 PDG code of incoming parton 1
   * \param[in] kf2 PDG code of incoming parton 2
   * \param[in] isub Subprocess identifier
   * \param[in] shat Center-of-mass energy squared [GeV²]
   * \param[in] Q2 Renormalization scale [GeV²]
   * \param[out] that_sampled Sampled Mandelstam t-hat [GeV²]
   * \param[out] phi_sampled Sampled azimuthal angle [radians]
   * \param[out] isub_sampled Selected subprocess ID (may differ from requested isub)
   * \return Total cross-section value
   * \throw std::runtime_error if no valid subprocess found
   */
  double sampleXS(int kf1, int kf2, int isub, double shat, double Q2,
                  double& that_sampled, double& phi_sampled, int& isub_sampled);

  /**
   * \brief Determine final state particles from subprocess
   * 
   * Maps a subprocess identifier to the PDG codes of outgoing partons.
   * Handles both elastic and inelastic final states.
   * 
   * \param[in] isub Subprocess identifier (1-10)
   * \param[in] kf1 PDG code of incoming parton 1
   * \param[in] kf2 PDG code of incoming parton 2
   * \param[out] kf3 PDG code of outgoing parton 1
   * \param[out] kf4 PDG code of outgoing parton 2
   */
  void getSubprocessFinalState(int isub, int kf1, int kf2,
                               int& kf3, int& kf4);

  /**
   * \brief Compute outgoing 4-momenta from scattering kinematics
   * 
   * Computes outgoing parton 4-momenta in lab frame given total initial
   * 4-momentum and sampled kinematic variables (t-hat and azimuthal angle).
   * Uses proper Lorentz boost from CM frame.
   * 
   * \param[in] p_total Total 4-momentum of incoming partons (E, px, py, pz) [GeV]
   * \param[in] that Sampled Mandelstam t-hat [GeV²]
   * \param[in] phi Sampled azimuthal angle [radians]
   * \param[in] shat Center-of-mass energy squared [GeV²]
   * \param[out] p3 Four-momentum of outgoing parton 1 [GeV]
   * \param[out] p4 Four-momentum of outgoing parton 2 [GeV]
   */
  void computeOutgoing4Momenta(const std::array<double, 4>& p_total,
                               double that, double phi, double shat,
                               std::array<double, 4>& p3,
                               std::array<double, 4>& p4);

  /**
   * \brief Get electric charge of a parton
   * 
   * Returns the fractional electric charge of a parton in units of
   * the elementary charge e.
   * 
   * \param[in] kf1 PDG code of the parton
   * \return Electric charge (e.g., -1/3 for down quark, 2/3 for up quark)
   */
  double partonCharge(int kf1);

  /**
   * \brief Compute t-hat limits for massless partons
   * 
   * Determines kinematic boundaries for t-hat including optional
   * minimum transverse momentum cut.
   * 
   * \param[in] s Center-of-mass energy squared [GeV²]
   * \param[in] pTmin Minimum transverse momentum cut [GeV]
   * \return TLimits struct with validated boundaries
   */
  static TLimits computeTHatLimitsMassless(double s, double pTmin = 0.0);

  /// Solver status: true if bmstofxi converged without exceeding iterations
  static bool lastBmstofxiOk;

  /// Solver residual: absolute difference \f$ |\xi_{\text{back}} - \xi_{\text{target}}| \f$
  static double lastBmstofxiResidual;

  /// Debug output level (0=off, 1+=basic, 2+=detailed)
  static int debugLevel;

  /**
   * \brief Reset solver status variables
   * 
   * Clears lastBmstofxiOk and lastBmstofxiResidual for new solver invocation.
   */
  static void resetLastBmstofxiStatus();
};

}  // namespace smash

#endif  // SRC_INCLUDE_SMASH_PARTONICCS_H_
