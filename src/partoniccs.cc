/*
 *    Copyright (C) 2026
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 */

#include "smash/partoniccs.h"

#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>

namespace smash {

int npxs = 200;
int nth = 200;
int intdim = 5 * nth;

std::vector<std::array<int, 3>> GlobalVar::isig(100);
std::vector<double> GlobalVar::FTH((nth + 1) * npxs);
std::vector<double> GlobalVar::FTH2((intdim + 1) * npxs);
std::vector<double> GlobalVar::VTH((nth + 1) * npxs);
std::vector<double> GlobalVar::VTH2((intdim + 1) * npxs);
std::vector<double> GlobalVar::TTH((nth + 1) * npxs);
std::vector<double> GlobalVar::TTH2((intdim + 1) * npxs);
std::vector<double> GlobalVar::xsec(npxs + 1);
std::vector<double> GlobalVar::sigh(100);

bool PartonicCS::lastBmstofxiOk = true;
double PartonicCS::lastBmstofxiResidual = 0.0;
int PartonicCS::debugLevel = 0;

/**
 * \brief Access coarse-grid integral: FTH[jj, ii]
 * 
 * Flattens 2D array access for FTH vector.
 */
double& getFTH(int jj, int ii) {
  return GlobalVar::FTH[jj * npxs + ii];
}

/**
 * \brief Access fine-grid integral: FTH2[jj, ii]
 * 
 * Flattens 2D array access for FTH2 vector.
 */
double& getFTH2(int jj, int ii) {
  return GlobalVar::FTH2[jj * npxs + ii];
}

/**
 * \brief Access coarse-grid xi: VTH[jj, ii]
 * 
 * Flattens 2D array access for VTH vector.
 */
double& getVTH(int jj, int ii) {
  return GlobalVar::VTH[jj * npxs + ii];
}

/**
 * \brief Access fine-grid xi: VTH2[jj, ii]
 * 
 * Flattens 2D array access for VTH2 vector.
 */
double& getVTH2(int jj, int ii) {
  return GlobalVar::VTH2[jj * npxs + ii];
}

/**
 * \brief Access coarse-grid t-hat: TTH[jj, ii]
 * 
 * Flattens 2D array access for TTH vector.
 */
double& getTTH(int jj, int ii) {
  return GlobalVar::TTH[jj * npxs + ii];
}

/**
 * \brief Access fine-grid t-hat: TTH2[jj, ii]
 * 
 * Flattens 2D array access for TTH2 vector.
 */
double& getTTH2(int jj, int ii) {
  return GlobalVar::TTH2[jj * npxs + ii];
}

/**
 * \brief Compute running strong coupling constant α_s
 * 
 * Evaluates the strong coupling at leading order using the QCD
 * running formula \f$ \alpha_s(Q^2) = \frac{1}{\beta_0 \ln(Q^2/\Lambda_{\text{QCD}}^2)} \f$
 * where \f$ \beta_0 = (33 - 2n_f)/(12\pi) \f$.
 * 
 * \param[in] Q2 Renormalization scale squared [GeV²]
 * \param[in] LambdaQCD QCD scale parameter [GeV]
 * \param[in] nf Number of active quark flavors
 * \return Strong coupling constant α_s at scale Q²
 * \throw std::invalid_argument if Q² ≤ 0
 * \throw std::domain_error if Q² ≤ Λ²_QCD (non-perturbative region)
 */
double PartonicCS::alphas(double Q2, double LambdaQCD, int nf) {
  if (Q2 <= 0.0) {
    throw std::invalid_argument("Q^2 must be positive.");
  }

  const double beta0 = (33.0 - 2.0 * nf) / (12.0 * M_PI);
  double logArg = Q2 / (LambdaQCD * LambdaQCD);

  if (logArg <= 1.0) {
    throw std::domain_error(
        "Q^2 is too close to or below Λ_QCD^2; non-perturbative region.");
  }

  double res = 1.0 / (beta0 * std::log(logArg));
  return res;
}

/**
 * \brief Get electric charge of a parton
 * 
 * Returns the fractional electric charge in units of the elementary charge.
 * Quark charges: d, s, b = -1/3; u, c = +2/3; gluon, photon = 0.
 * 
 * \param[in] kf1 PDG code of the parton
 * \return Electric charge (units of e)
 */
double PartonicCS::partonCharge(int kf1) {
  int kf = std::abs(kf1);
  double ei = 0.0;
  if (kf == 1 || kf == 3 || kf == 5) {
    ei = -1.0 / 3.0;
  } else if (kf == 2 || kf == 4) {
    ei = 2.0 / 3.0;
  } else if (kf == 21) {
    ei = 0.0;
  } else if (kf == 22) {
    ei = 0.0;
  } else {
    ei = 0.0;
  }
  return ei;
}

/**
 * \brief Compute differential cross-section for a 2 → 2 subprocess
 * 
 * Calculates \f$ d\sigma/dt \f$ for the given subprocess using
 * QCD tree-level matrix elements. Supports 10 different subprocesses:
 * 1. q + q' → q + q'
 * 2. q + q̄ → q' + q̄'
 * 3. q + q̄ → g + g
 * 4. q + q̄ → g + γ
 * 5. q + q̄ → γ + γ
 * 6. q + g → q + g
 * 7. q + g → q + γ
 * 8. g + g → q + q̄
 * 9. g + g → g + g
 * 10. Soft (low-pT) phenomenological ansatz
 * 
 * \param[in] isub Subprocess identifier (1-10)
 * \param[in] kf1 PDG code of incoming parton 1
 * \param[in] kf2 PDG code of incoming parton 2
 * \param[in] shat Mandelstam variable \f$ \hat{s} \f$ [GeV²]
 * \param[in] that Mandelstam variable \f$ \hat{t} \f$ [GeV²]
 * \param[in] Q2 Renormalization scale \f$ Q^2 \f$ [GeV²]
 * \param[out] nchn Number of matrix element channels contributing
 * \param[out] dsigdt Result: differential cross-section [GeV⁻²]
 * \throw std::runtime_error if isub is not in valid range
 */
void PartonicCS::clox(int isub, int kf1, int kf2, double shat, double that,
                      double Q2, int& nchn, double& dsigdt) {
    dsigdt = 0.0;
    if (isub >= 0 && static_cast<size_t>(isub) < GlobalVar::xsec.size()) {
        GlobalVar::xsec[isub] = 0.0;
    }
    nchn = 0;
    
    double SQM1 = 0.0, SQM2 = 0.0;
    double SQM3 = 0.0, SQM4 = 0.0;
    double uhat = -that - shat + SQM1 + SQM2 + SQM3 + SQM4;
    int ns = nchn;
    double as = PartonicCS::alphas(Q2);
    double faca = 1.0;
    double aem = 1.0 / 137.0;
    double comfac = M_PI / (shat * shat);
    
    switch (isub) {
        case 1: {
            double facqq1 = comfac * as * as * 4.0 / 9.0 * (shat * shat + uhat * uhat) / (that * that);
            double facqqb = comfac * as * as * 4.0 / 9.0 * ((shat * shat + uhat * uhat) / (that * that) * faca -
                                                            mstp34 * 2.0 / 3.0 * uhat * uhat / (shat * that));
            double facqq2 = comfac * as * as * 4.0 / 9.0 * ((shat * shat + that * that) / (uhat * uhat) -
                                                            mstp34 * 2.0 / 3.0 * shat * shat / (that * uhat));
            
            nchn++;
            GlobalVar::isig[nchn] = {kf1, kf2, 1};
            GlobalVar::sigh[nchn] = std::max(0.0, facqq1);
            if (kf1 == -kf2) {
                GlobalVar::sigh[nchn] = std::max(0.0, facqqb);
            }
            if (kf1 == kf2) {
                nchn++;
                GlobalVar::isig[nchn] = {kf1, kf2, 2};
                GlobalVar::sigh[nchn] = std::max(0.0, facqq2);
            }

            for (int i = ns + 1; i <= nchn; ++i) {
                GlobalVar::xsec[1] += GlobalVar::sigh[i];
                dsigdt += GlobalVar::sigh[i];
            }
            break;
        }
            
        case 2: {
            double facqqb = comfac * as * as * 4.0 / 9.0 * (that * that + uhat * uhat) / (shat * shat);
            nchn++;
            GlobalVar::isig[nchn] = {kf1, kf2, 1};
            GlobalVar::sigh[nchn] = std::max(0.0, facqqb);
            
            for (int i = ns + 1; i <= nchn; ++i){
                GlobalVar::xsec[2] += GlobalVar::sigh[i];
                dsigdt += GlobalVar::sigh[i];
            }
            break;
        }
            
        case 3: {
            double factor = 32.0 / 27.0;
            double facgg1 = comfac * as * as * factor * (uhat / that -
                                                         (2.0 + mstp34 * 0.25) * uhat * uhat / (shat * shat));
            double facgg2 = comfac * as * as * factor * (that / uhat -
                                                         (2.0 + mstp34 * 0.25) * that * that / (shat * shat));
            
            nchn++;
            GlobalVar::isig[nchn] = {kf1, kf2, 1};
            GlobalVar::sigh[nchn] = std::max(0.0, facgg1);
            nchn++;
            GlobalVar::isig[nchn] = {kf1, kf2, 2};
            GlobalVar::sigh[nchn] = std::max(0.0, facgg2); 

            for (int i = ns + 1; i <= nchn; ++i) {
                GlobalVar::xsec[3] += GlobalVar::sigh[i];
                dsigdt += GlobalVar::sigh[i];
            }
            break;
        }
            
        case 4: {
            double facgg = comfac * as * aem * 8.0 / 9.0 * (that / uhat + uhat / that);
            double ei = PartonicCS::partonCharge(kf1);
            nchn++;
            GlobalVar::isig[nchn] = {kf1, kf2, 1};
            GlobalVar::sigh[nchn] = std::max(0.0, facgg) * ei * ei;

            for (int i = ns + 1; i <= nchn; ++i){
                GlobalVar::xsec[4] += GlobalVar::sigh[i];
                dsigdt += GlobalVar::sigh[i];
            }
            break;
        }
            
        case 5: {
            double facgg = comfac * aem * aem * 2.0 * (that / uhat + uhat / that);
            double ei = PartonicCS::partonCharge(kf1);
            double fcoi = (std::abs(kf1) <= 10) ? faca / 3.0 : 1.0;
            
            nchn++;
            GlobalVar::isig[nchn] = {kf1, kf2, 1};
            GlobalVar::sigh[nchn] = std::max(0.0, facgg) * fcoi * pow(ei, 4);

            for (int i = ns + 1; i <= nchn; ++i){
                GlobalVar::xsec[5] += GlobalVar::sigh[i];
                dsigdt += GlobalVar::sigh[i];
            }
            break;
        }    
            
        case 6: {
            if(kf1 == 21) {
                kf1 = kf2; kf2 = 21;
            }
            
            double facqg1 = comfac * as * as * 4.0 / 9.0 *((2.0 + mstp34 * 0.25) * uhat * uhat / (that * that) - uhat / shat) * faca;
            double facqg2 = comfac * as * as * 4.0 / 9.0 *((2.0 + mstp34 * 0.25) * shat * shat / (that * that) - shat / uhat);
            
            for (int j=0; j < 2; j++) {
                nchn++;
                GlobalVar::isig[nchn][j] = kf1;
                GlobalVar::isig[nchn][2 - j] = 21;
                GlobalVar::isig[nchn][2] = 1;
                GlobalVar::sigh[nchn] = 0.5 * std::max(0.0, facqg1);

                nchn += 1;
                GlobalVar::isig[nchn][j] = kf1;
                GlobalVar::isig[nchn][2 - j] = 21;
                GlobalVar::isig[nchn][2] = 2;
                GlobalVar::sigh[nchn] = 0.5 * std::max(0.0, facqg2);
            }
            for (int i = ns + 1; i <= nchn; ++i){
                GlobalVar::xsec[6] += GlobalVar::sigh[i];
                dsigdt += GlobalVar::sigh[i];
            }
            break;
        }
            
        case 7: {
            if (kf1 == 21) {
                    std::swap(kf1, kf2);
            }
            double fgq = -comfac * faca * as * aem * (1.0 / 3.0) * (shat / uhat + uhat / shat);
            double ei = PartonicCS::partonCharge(kf1);
            double facgq = fgq * ei * ei;

            for (int isde = 1; isde <= 2; ++isde) {
                ns = nchn;
                nchn++;
                GlobalVar::isig[nchn][isde] = kf1;
                GlobalVar::isig[nchn][3 - isde] = 21;
                GlobalVar::isig[nchn][3] = 1;
                GlobalVar::sigh[nchn] = std::max(0.0, facgq);
                for (int i = ns + 1; i <= nchn; ++i) {
                    GlobalVar::xsec[7] += GlobalVar::sigh[i];
                    dsigdt += GlobalVar::sigh[i];
                }
            }
            break;
        }
            
        case 8: {
            double facqq1 = comfac * as * as * (1.0 / 6.0) * (uhat / that - (2.0 + mstp34 * 0.25) * uhat * uhat / (shat * shat)) * faca;
            double facqq2 = comfac * as * as * (1.0 / 6.0) * (that / uhat - (2.0 + mstp34 * 0.25) * that * that / (shat * shat)) * faca;
            ns = nchn;
            nchn++;
            GlobalVar::isig[nchn][1] = 21;
            GlobalVar::isig[nchn][2] = 21;
            GlobalVar::isig[nchn][3] = 1;
            GlobalVar::sigh[nchn] = std::max(0.0, facqq1);
            nchn++;
            GlobalVar::isig[nchn][1] = 21;
            GlobalVar::isig[nchn][2] = 21;
            GlobalVar::isig[nchn][3] = 2;
            GlobalVar::sigh[nchn] = std::max(0.0, facqq2);

            for (int i = ns + 1; i <= nchn; ++i) {
                GlobalVar::xsec[8] += GlobalVar::sigh[i];
                dsigdt += GlobalVar::sigh[i];
            }
            break;
        }
            
        case 9: {
            double facgg1 = comfac * as * as * 9.0 / 4.0 * uhat * uhat * (1.0 / (that * that) + 1.0 / (shat * shat) + 1.0 / (uhat * uhat)) * faca;
            double facgg2 = comfac * as * as * 9.0 / 4.0 * (that * that) * (1.0 / (that * that) + 1.0 / (shat * shat) + 1.0 / (uhat * uhat)) * faca;
            double facgg3 = comfac * as * as * 9.0 / 4.0 * shat * shat * (1.0 / (that * that) + 1.0 / (shat * shat) + 1.0 / (uhat * uhat));
            ns = nchn;
            nchn++;
            GlobalVar::isig[nchn][1] = 21;
            GlobalVar::isig[nchn][2] = 21;
            GlobalVar::isig[nchn][3] = 1;
            GlobalVar::sigh[nchn] = std::max(0.0, facgg1);
            nchn++;
            GlobalVar::isig[nchn][1] = 21;
            GlobalVar::isig[nchn][2] = 21;
            GlobalVar::isig[nchn][3] = 2;
            GlobalVar::sigh[nchn] = std::max(0.0, facgg2);
            nchn++;
            GlobalVar::isig[nchn][1] = 21;
            GlobalVar::isig[nchn][2] = 21;
            GlobalVar::isig[nchn][3] = 3;
            GlobalVar::sigh[nchn] = std::max(0.0, facgg3);

            for (int i = ns + 1; i <= nchn; ++i) {
                GlobalVar::xsec[9] += GlobalVar::sigh[i];
                dsigdt += GlobalVar::sigh[i];
            }
            break;
        }
        
        case 10: {
            double parv48 = 0.5;
            double sig0 = 9.0 / 2.0 * M_PI * as * as / pow((Q2 + parv48), 2);
            nchn++;
            GlobalVar::isig[nchn][1] = kf1;
            GlobalVar::isig[nchn][2] = kf2;
            GlobalVar::isig[nchn][3] = 1;
            GlobalVar::sigh[nchn] = sig0;

            GlobalVar::xsec[10] += GlobalVar::sigh[nchn];
            dsigdt += GlobalVar::sigh[nchn];
            break;
        }
            
        default: {
            std::ostringstream ss;
            ss << "clox: isub value not implemented - isub=" << isub
               << " kf1=" << kf1 << " kf2=" << kf2
               << " shat=" << shat << " that=" << that << " Q2=" << Q2;
            throw std::runtime_error(ss.str());
        }
            break;
    }
}

/**
 * \brief Transform Mandelstam variable to xi coordinate
 * 
 * Computes the integration variable xi and its Jacobian \f$ d\hat{t}/d\xi \f$
 * for a given subprocess. The transformation is subprocess-specific and chosen
 * to optimize numerical integration properties.
 * 
 * \param[in] isub Subprocess identifier
 * \param[in] kf1 PDG code of incoming parton 1
 * \param[in] kf2 PDG code of incoming parton 2
 * \param[in] shat Mandelstam variable \f$ \hat{s} \f$ [GeV²]
 * \param[in] that Mandelstam variable \f$ \hat{t} \f$ [GeV²]
 * \param[out] dtdxi Jacobian \f$ d\hat{t}/d\xi \f$
 * \param[out] xi Transformed variable value
 * \throw std::runtime_error if isub is not implemented
 */
void PartonicCS::bmsxi(int isub, int kf1, int kf2, double shat, double that,
                       double& dtdxi, double& xi) {
    double SQM1 = 0.0, SQM2 = 0.0;
    double SQM3 = 0.0, SQM4 = 0.0;
    double uhat = -that - shat + SQM1 + SQM2 + SQM3 + SQM4;

    double amc, del, zz, tt, f2, f3, f4;

    switch (isub) {
    case 1: {
        if (kf1 == -kf2) {
            dtdxi = that * that / (shat * shat);
            xi = -shat * shat / that;
        } else if (kf1 == kf2) {
            dtdxi = (that * that * uhat * uhat) / (shat * shat * shat);
            xi = 2.0 * std::log((shat + that) / -that) - shat / that - shat / (shat + that);
        } else {
            dtdxi = that * that / (shat * shat);
            xi = -shat * shat / that;
        }
        break;
    }

    case 2: case 11: case 12: case 13: {
        dtdxi = 1.0;
        xi = that;
        break;
    }

    case 3: case 4: case 5: case 8: {
        dtdxi = that * uhat / (shat * shat);
        xi = shat * std::log(shat + that) - shat * std::log(-that);
        break;
    }

    case 6: {
        dtdxi = that * that * uhat / std::pow(shat, 3);
        xi = shat * shat / that + shat * std::log(that / uhat);
        break;
    }

    case 7: {
        dtdxi = uhat/shat;
        xi = -shat*std::log(-uhat);
        break;
    }

    case 9: {
        dtdxi = that * that * uhat * uhat / std::pow(shat, 3);
        xi = 2.0 * std::log((shat + that) / -that) - shat / that - shat / (shat + that);
        break;
    }

    case 10: {
        double parv48 = 0.5;
        dtdxi = std::pow(-that + parv48 * parv48, 2);
        xi = 1.0 / (-that + parv48 * parv48);
        break;
    }

    default: {
        std::ostringstream ss;
        ss << "bmsxi: isub value not implemented - isub=" << isub
           << " kf1=" << kf1 << " kf2=" << kf2
           << " shat=" << shat << " that=" << that;
        throw std::runtime_error(ss.str());
    }
    }
}

/**
 * \brief Inverse transformation: solve for t̂ given ξ
 * 
 * Uses an improved Newton method with bisection to find the t̂ value
 * corresponding to a target value of the xi variable. Robustly handles
 * non-finite values at boundaries and failed evaluations.
 * 
 * \param[in] isub Subprocess identifier
 * \param[in] kf1 PDG code of incoming parton 1
 * \param[in] kf2 PDG code of incoming parton 2
 * \param[in] shat Mandelstam variable \f$ \hat{s} \f$ [GeV²]
 * \param[in] xi_target Target value of xi
 * \param[in] tmin Lower kinematic boundary [GeV²]
 * \param[in] tmax Upper kinematic boundary [GeV²]
 * \return t̂ value at which bmsxi gives xi_target
 * 
 * Sets \ref lastBmstofxiOk and \ref lastBmstofxiResidual for diagnostics.
 */
double PartonicCS::bmstofxi(int isub, int kf1, int kf2, double shat,
                            double xi_target, double tmin, double tmax) {
    double dtdxi_min=0.0, xi_min=0.0, dtdxi_max=0.0, xi_max=0.0;
    bmsxi(isub,kf1,kf2,shat,tmin,dtdxi_min,xi_min);

  if (!std::isfinite(xi_min)) {
    const int Nscan = 256;
    for (int is = 1; is <= Nscan; ++is) {
      double tt = tmin + (tmax - tmin) * (static_cast<double>(is) /
                                           static_cast<double>(Nscan));
      double dtdtmp = 0.0, xitmp = 0.0;
      bool ok = true;
      try {
        bmsxi(isub, kf1, kf2, shat, tt, dtdtmp, xitmp);
      } catch (...) {
        ok = false;
      }
      if (!ok) continue;
      if (std::isfinite(xitmp)) {
        tmin = tt;
        dtdxi_min = dtdtmp;
        xi_min = xitmp;
        break;
      }
    }
  }

    double fl = xi_min - xi_target;
    if (std::abs(xi_min) != 0.0 && std::abs(fl/xi_min) < PartonicCS::xacc) return tmin;

    bmsxi(isub,kf1,kf2,shat,tmax,dtdxi_max,xi_max);
  if (!std::isfinite(xi_max)) {
    const int Nscan = 256;
    for (int is = 1; is <= Nscan; ++is) {
      double tt = tmax - (tmax - tmin) * (static_cast<double>(is) /
                                           static_cast<double>(Nscan));
      double dtdtmp = 0.0, xitmp = 0.0;
      bool ok = true;
      try {
        bmsxi(isub, kf1, kf2, shat, tt, dtdtmp, xitmp);
      } catch (...) {
        ok = false;
      }
      if (!ok) continue;
      if (std::isfinite(xitmp)) {
        tmax = tt;
        dtdxi_max = dtdtmp;
        xi_max = xitmp;
        break;
      }
    }
  }

    double fh = xi_max - xi_target;
    if (std::abs(xi_max) != 0.0 && std::abs(fh/xi_max) < PartonicCS::xacc) return tmax;

  if (!std::isfinite(xi_min) || !std::isfinite(xi_max)) {
    double mid = 0.5 * (tmin + tmax);
    PartonicCS::lastBmstofxiOk = false;
    PartonicCS::lastBmstofxiResidual = NAN;
    return mid;
  }

    double xl = fl < 0.0 ? tmin : tmax;
    double xh = fl < 0.0 ? tmax : tmin;
    double that = 0.5*(tmin+tmax);

    double f=0.0, df_num=0.0, dtdxi=0.0, xi=0.0;
    PartonicCS::lastBmstofxiOk = true;
    PartonicCS::lastBmstofxiResidual = 1e300;

    if ((fl > 0.0 && fh > 0.0) || (fl < 0.0 && fh < 0.0)){
        const int Nscan = 1024;
        double prev_t = tmin;
        double prev_xi = xi_min;
        double prev_f = fl;
        bool found = false;
        for (int is=1; is<=Nscan; ++is){
            double tt = tmin + (tmax - tmin) * (static_cast<double>(is)/static_cast<double>(Nscan));
            double dtdtmp=0.0, xitmp=0.0;
            bool ok = true;
            try{ bmsxi(isub,kf1,kf2,shat,tt,dtdtmp,xitmp); } catch(...) { ok = false; }
            if (!ok) {
          prev_t = tt;
          prev_xi = xitmp;
          prev_f = xitmp - xi_target;
          continue;
        }
        if (!std::isfinite(xitmp)) {
          prev_t = tt;
          prev_xi = xitmp;
          prev_f = xitmp - xi_target;
          continue;
        }
            double fi = xitmp - xi_target;
        if (std::isfinite(prev_f) && (prev_f * fi <= 0.0)) {
          xl = prev_t;
          xh = tt;
          that = 0.5 * (xl + xh);
          found = true;
          break;
        }
            prev_t = tt; prev_xi = xitmp; prev_f = fi;
        }
    if (!found) {
      if (std::abs(fl) <= std::abs(fh)) {
        xl = tmin;
        xh = tmax;
        that = tmin;
      } else {
        xl = tmin;
        xh = tmax;
        that = tmax;
      }
    }
    }

  for (int it = 0; it < PartonicCS::maxit; ++it) {
    bmsxi(isub, kf1, kf2, shat, that, dtdxi, xi);
    f = xi - xi_target;

    const double min_dtdxi = 1e-8;
    if (std::abs(dtdxi) < min_dtdxi) {
      that = 0.5 * (xl + xh);
    } else {
      df_num = 1.0 / dtdxi;
      double dx = f / df_num;
      double max_step = 0.25 * (xh - xl);
      if (std::abs(dx) > max_step) dx = std::copysign(max_step, dx);
      double that_new = that - dx;

      if ((that_new <= xl) || (that_new >= xh)) {
        that = 0.5 * (xl + xh);
      } else {
        double dtdxi_trial = 0.0, xi_trial = 0.0;
        bool trial_ok = true;
        try {
          bmsxi(isub, kf1, kf2, shat, that_new, dtdxi_trial, xi_trial);
        } catch (...) {
          trial_ok = false;
        }
        if (!trial_ok) {
          that = 0.5 * (xl + xh);
          goto update_bracket;
        }
        double f_trial = xi_trial - xi_target;
        if (!std::isfinite(f_trial) || std::abs(f_trial) >= std::abs(f)) {
          that = 0.5 * (xl + xh);
        } else {
          that = that_new;
        }
      }
    }

    if (std::abs(0.5 * (xh - xl)) < PartonicCS::xacc) {
      double dtdxi_f = 0.0, xi_f = 0.0;
      bmsxi(isub, kf1, kf2, shat, that, dtdxi_f, xi_f);
      PartonicCS::lastBmstofxiResidual = std::abs(xi_f - xi_target);
      PartonicCS::lastBmstofxiOk = true;
      return that;
    }

  update_bracket:
    bmsxi(isub, kf1, kf2, shat, that, dtdxi, xi);
    f = xi - xi_target;
    if (f < 0.0) {
      xl = that;
    } else {
      xh = that;
    }
    }

  double mid = 0.5 * (xl + xh);
  double dtdxi_f = 0.0, xi_f = 0.0;
  bmsxi(isub, kf1, kf2, shat, mid, dtdxi_f, xi_f);
  PartonicCS::lastBmstofxiResidual = std::abs(xi_f - xi_target);
  PartonicCS::lastBmstofxiOk = false;
  return mid;
}

/**
 * \ Compute t-hat limits for massless partons with pT cut
 * 
 * Calculates kinematic boundaries for t̂ including an optional
 * minimum transverse momentum cut. The result is slightly inset
 * from the exact boundaries to avoid numerical singularities.
 * 
 * \param[in] s Center-of-mass energy squared [GeV²]
 * \param[in] pTmin Minimum transverse momentum cut [GeV]
 * \return TLimits struct with validated boundaries and ok flag
 */
TLimits PartonicCS::computeTHatLimitsMassless(double s, double pTmin) {
  TLimits out{0.0, 0.0, false};
  if (s <= 0.0) return out;
  double th_min = -s;
  double th_max = 0.0;

  if (pTmin > 0.0) {
    double pt2 = pTmin * pTmin;
    if (4.0 * pt2 >= s) return out;
    double root = std::sqrt(std::max(0.0, 1.0 - 4.0 * pt2 / s));
    double t_cut = -0.5 * s * (1.0 - root);
    th_max = t_cut;
    th_min = -s - t_cut;
  }

  double span = th_max - th_min;
  double eps = std::max(1e-14 * std::abs(span), 1e-12);
  th_min += eps;
  th_max -= eps;
  if (th_max <= th_min) return out;
  out.tmin = th_min;
  out.tmax = th_max;
  out.ok = true;
  return out;
}

/**
 * \ Reset solver status variables
 * 
 * Clears diagnostic variables set by bmstofxi for a fresh solver run.
 */
void PartonicCS::resetLastBmstofxiStatus() {
  PartonicCS::lastBmstofxiOk = true;
  PartonicCS::lastBmstofxiResidual = 1e300;
}

/**
 * \brief Sample collisional cross-section and parton kinematics
 * 
 * Computes the total partonic cross-section for a collision, selects a
 * subprocess according to the cross-section distribution, and generates
 * the kinematics (t̂, φ) of outgoing partons. Uses coarse and fine
 * integration grids for robust sampling.
 * 
 * \param[in] kf1 PDG code of incoming parton 1
 * \param[in] kf2 PDG code of incoming parton 2
 * \param[in] isub Subprocess identifier (1-10)
 * \param[in] shat Center-of-mass energy squared \f$ \hat{s} \f$ [GeV²]
 * \param[in] Q2 Renormalization scale [GeV²]
 * \return Total cross-section value [GeV⁻²]
 * \throw std::runtime_error if no valid subprocess found in sampling
 */
double PartonicCS::sampleXS(int kf1, int kf2, int isub, double shat,
                            double Q2,
                            double& that_sampled, double& phi_sampled,
                            int& isub_sampled) {
    int npro = 0;
    std::vector<int> ipro(npxs + 1, 0);
    for (int ii = 0; ii < npxs; ++ii) {
        GlobalVar::xsec[ii + 1] = 0.0;
    }
    
    if (std::abs(kf1) > 6 && kf1 != 21) {
        return 0.0;
    }
    if (std::abs(kf2) > 6 && kf2 != 21) {
        return 0.0;
    }
    
    if (kf1 != 21 && kf2 != 21) {
        if (kf1 * kf2 > 0) {
            ipro[npro++] = 1;
        } else if (kf1 * kf2 < 0) {
            if (kf1 != -kf2) {
                ipro[npro++] = 1;
            } else {
                    for (int ii = 1; ii <= 5; ++ii) {
                        ipro[npro++] = ii;
                    }
            }
        }
    } else if (kf1 != 21 || kf2 != 21) {
        for (int ii = 6; ii <= 7; ++ii) {
            ipro[npro++] = ii;
        }
    } else if (kf1 == 21 && kf2 == 21) {
        for (int ii = 8; ii <= 9; ++ii) {
            ipro[npro++] = ii;
        }
    }
    
    if (npro == 0) {
        return 0.0;
    }
    
    int kmin = 1;
    int kmax = npro;
    for (int kpro = kmin - 1; kpro < kmax; ++kpro) {
        isub = ipro[kpro];
        double shatmin = 1.0;
        double pthmin = 0.5;
        
        if (shat < std::max<double>(shatmin, 4.0 * std::pow(pthmin, 2))) {
            continue;
        }
        
        double thl = -shat;
        double thu = 0.0;
        TLimits lim = PartonicCS::computeTHatLimitsMassless(shat, pthmin);
        if (lim.ok) {
            thl = lim.tmin;
            thu = lim.tmax;
        } else {
            GlobalVar::xsec[isub] = 0.0;
            continue;
        }
        
        double thi = (thu - thl) / static_cast<double>(nth);
        double dtdxi_mn, ximin, dtdxi_mx, ximax;
        bmsxi(isub, kf1, kf2, shat, thl, dtdxi_mn, ximin);
        bmsxi(isub, kf1, kf2, shat, thu, dtdxi_mx, ximax);
        
        if (!std::isfinite(ximin)){
            const int Nscan = 256;
            for (int is=1; is<=Nscan; ++is){
                double tt = thl + (thu - thl) * (static_cast<double>(is)/static_cast<double>(Nscan));
                double dtdtmp=0.0, xitmp=0.0;
                try { bmsxi(isub, kf1, kf2, shat, tt, dtdtmp, xitmp); } catch(...) { continue; }
                if (std::isfinite(xitmp)) { ximin = xitmp; break; }
            }
        }
        if (!std::isfinite(ximax)){
            const int Nscan = 256;
            for (int is=1; is<=Nscan; ++is){
                double tt = thu - (thu - thl) * (static_cast<double>(is)/static_cast<double>(Nscan));
                double dtdtmp=0.0, xitmp=0.0;
                try { bmsxi(isub, kf1, kf2, shat, tt, dtdtmp, xitmp); } catch(...) { continue; }
                if (std::isfinite(xitmp)) { ximax = xitmp; break; }
            }
        }
        if (!std::isfinite(ximin) || !std::isfinite(ximax)){
            GlobalVar::xsec[isub] = 0.0;
            continue;
        }

        double dxi = (ximax - ximin) / static_cast<double>(nth);

        getFTH(0, isub) = 0.0;
        getVTH(0, isub) = ximin;

        for (int ii = 1; ii <= nth; ++ii) {
            getVTH(ii, isub) = ximin + ii * dxi;
            double that = bmstofxi(isub, kf1, kf2, shat, getVTH(ii, isub), thl, thu);
            getTTH(ii, isub) = that;

            double colxs = 0.0;
            int nchn = 0;
            try {
                PartonicCS::clox(isub, kf1, kf2, shat, that, Q2, nchn, colxs);
            } catch (...) {
            }

            double dtdxi = 0.0, xi = 0.0;
            try {
              bmsxi(isub, kf1, kf2, shat, that, dtdxi, xi);
            } catch (...) {
              dtdxi = NAN;
            }

            double deltaV = getVTH(ii, isub) - getVTH(ii - 1, isub);
  if (std::isfinite(colxs) && std::isfinite(dtdxi) &&
      std::isfinite(deltaV)) {
    getFTH(ii, isub) =
        getFTH(ii - 1, isub) + colxs * dtdxi * deltaV;
            } else {
                getFTH(ii, isub) = getFTH(ii - 1, isub);
            }
        }
    }
    
    double sigtot = 0.0, sigfac = 1.0;
    for (int kpro = kmin - 1; kpro < kmax; ++kpro) {
        isub = ipro[kpro];

        if ((isub == 1 && kf1 == kf2) || isub == 8 || isub == 9 || isub == 16 || isub == 17) {
            for (int i = 0; i <= nth; ++i) {
                getFTH(i, isub) *= 0.5;
            }
        }

        GlobalVar::xsec[isub] = sigfac * getFTH(nth, isub);
        sigtot += GlobalVar::xsec[isub];
    }

    double colxs = sigtot;

    if (sigtot <= 1.0e-8) {
        return 0.0;
    }
    
    double rint = std::max(0.0, colxs) / M_PI;
    if (rint == 0.0) {
        return 0.0;
    }

    for (int ii = 0; ii < npxs; ++ii) {
        for (int jj = 0; jj <= intdim; ++jj) {
            getFTH2(jj, ii + 1) = 0.0;
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    int nsub = 30;
    double distrib = dist(gen);
    double rsub = colxs * distrib;
    for (int i = 1; i <= nsub; ++i) {
        isub = i;
        rsub -= GlobalVar::xsec[i];
        if (rsub <= 0.0) break;
    }
    if (rsub > 0.0) {
        throw std::runtime_error("No valid subprocess found");
    }
    
    double thl = -shat;
    double thu = 0.0;
    double pthmin = 1.0;
    
    TLimits lim = PartonicCS::computeTHatLimitsMassless(shat, pthmin);
    if (lim.ok) {
        thl = lim.tmin;
        thu = lim.tmax;
    } else {
        GlobalVar::xsec[isub] = 0.0;
    }   

    double thi = (thu - thl) / static_cast<double>(intdim);
    double dtdxi_mn, ximin, dtdxi_mx, ximax;
    bmsxi(isub, kf1, kf2, shat, thl, dtdxi_mn, ximin);
    bmsxi(isub, kf1, kf2, shat, thu, dtdxi_mx, ximax);
    double dxi = (ximax - ximin) / static_cast<double>(intdim);

    getFTH2(0, isub) = 0.0;
    getVTH2(0, isub) = ximin;
    getTTH2(0, isub) = thl;

    for (int ii = 1; ii <= intdim; ++ii) {
        getVTH2(ii, isub) = ximin + ii * dxi;
        double that = bmstofxi(isub, kf1, kf2, shat, getVTH2(ii, isub), thl, thu);
        getTTH2(ii, isub) = that;

        double colxs = 0.0;
        int nchn = 0;
        try {
            PartonicCS::clox(isub, kf1, kf2, shat, that, Q2, nchn, colxs);
        } catch (...) {
            colxs = 0.0;
        }

        double dtdxi = 0.0, xi = 0.0;
        try {
            bmsxi(isub, kf1, kf2, shat, that, dtdxi, xi);
        } catch (...) {
            dtdxi = NAN;
        }

        double deltaV = getVTH2(ii, isub) - getVTH2(ii - 1, isub);
        if (std::isfinite(colxs) && std::isfinite(dtdxi) && std::isfinite(deltaV)) {
            getFTH2(ii, isub) = getFTH2(ii - 1, isub) + colxs * dtdxi * deltaV;
        } else {
            getFTH2(ii, isub) = getFTH2(ii - 1, isub);
        }
    }
    
    if ((isub == 1 && kf1 == kf2) || isub == 8 || isub == 9 || isub == 16 || isub == 17) {
        for (int i = 0; i <= intdim; ++i) {
            getFTH2(i, isub) *= 0.5;
        }
    }
    
    std::random_device rd1;
    std::mt19937 gen1(rd1());
    std::uniform_real_distribution<> dist1(0.0, 1.0);
    
    double rdth = dist1(gen1) * getFTH2(intdim, isub);
    int i=0;
    for(; i<intdim; ++i) {
        if(rdth <= getFTH2(i+1, isub)) break;
    }
    double dfdt = (getFTH2(i + 1, isub) - getFTH2(i, isub)) / (getTTH2(i + 1, isub) - getTTH2(i, isub));
    double deltaf = rdth - getFTH2(i, isub);
    double that = getTTH2(i, isub) + deltaf / dfdt;
    
    if (that < thl || that > thu) {
        throw std::runtime_error("TH out of bounds");
    }
    
    double dtdxi, xi;
    bmsxi(isub, kf1, kf2, shat, that, dtdxi, xi);
    
    that = std::max(thl, std::min(thu, that));
    double uhat = - shat - that;

    std::random_device rdphi;
    std::mt19937 genphi(rdphi());
    std::uniform_real_distribution<> distphi(0.0, 1.0); 
    double phi = M_PI * distphi(genphi);
    
    // Return sampled kinematics to caller
    that_sampled = that;
    phi_sampled = phi;
    isub_sampled = isub;
    
    return colxs;
}

void PartonicCS::getSubprocessFinalState(int isub, int kf1, int kf2,
                                           int& kf3, int& kf4) {
    // Determine final state particles based on subprocess
    switch (isub) {
        case 1:  // q + q → q + q or q + qbar → q + qbar (elastic)
            kf3 = kf1;
            kf4 = kf2;
            break;
        case 3:  // q + qbar → g + g
            kf3 = 21;
            kf4 = 21;
            break;
        case 6:  // q + g → q + g (elastic)
            kf3 = (kf1 != 21) ? kf1 : kf2;
            kf4 = 21;
            break;
        case 8:  // g + g → q + qbar (choose random flavor)
            {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_real_distribution<> dist(0.0, 1.0);
                int nflavor = 3;  // u, d, s
                int flavor = 1 + static_cast<int>(dist(gen) * nflavor);
                kf3 = flavor;
                kf4 = -flavor;
            }
            break;
        case 9:  // g + g → g + g (elastic)
            kf3 = 21;
            kf4 = 21;
            break;
        default:
            kf3 = kf1;
            kf4 = kf2;
            break;
    }
}

void PartonicCS::computeOutgoing4Momenta(const std::array<double, 4>& p_total,
                                        double that, double phi, double shat,
                                        std::array<double, 4>& p3,
                                        std::array<double, 4>& p4) {
    // Extract total momentum components
    double E_tot = p_total[0];
    double px_tot = p_total[1];
    double py_tot = p_total[2];
    double pz_tot = p_total[3];
    double sqrt_s = std::sqrt(std::max(0.0, shat));
    
    if (sqrt_s < 0.1) {  // Below threshold
        p3 = {0, 0, 0, 0};
        p4 = {0, 0, 0, 0};
        return;
    }
    
    // Energy of each outgoing parton in CM frame (massless)
    double Ecm = sqrt_s / 2.0;
    double pcm = Ecm;  // Momentum magnitude (massless partons)
    
    // Scattering angle from Mandelstam t
    // t = -2*p_cm^2*(1 - cos(theta))
    double cos_theta = 1.0 + that / (2.0 * pcm * pcm);
    cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    
    // Outgoing momentum in CM frame (parton 3 scatters, parton 4 recoils)
    double p3x_cm = pcm * sin_theta * std::cos(phi);
    double p3y_cm = pcm * sin_theta * std::sin(phi);
    double p3z_cm = pcm * cos_theta;
    
    double p4x_cm = -p3x_cm;
    double p4y_cm = -p3y_cm;
    double p4z_cm = -p3z_cm;
    
    // Boost back to lab frame
    double beta_x = px_tot / E_tot;
    double beta_y = py_tot / E_tot;
    double beta_z = pz_tot / E_tot;
    double beta_mag2 = beta_x*beta_x + beta_y*beta_y + beta_z*beta_z;
    double gamma = E_tot / sqrt_s;
    
    // Boost parton 3
    double E3_cm = Ecm;
    double p3_dot_beta = p3x_cm * beta_x + p3y_cm * beta_y + p3z_cm * beta_z;
    double boost_factor = (beta_mag2 > 1e-10) ? (gamma - 1.0) / beta_mag2 : 0.0;
    
    p3[0] = gamma * (E3_cm + p3_dot_beta);
    p3[1] = p3x_cm + boost_factor * p3_dot_beta * beta_x + gamma * beta_x * E3_cm;
    p3[2] = p3y_cm + boost_factor * p3_dot_beta * beta_y + gamma * beta_y * E3_cm;
    p3[3] = p3z_cm + boost_factor * p3_dot_beta * beta_z + gamma * beta_z * E3_cm;
    
    // Boost parton 4
    double E4_cm = Ecm;
    double p4_dot_beta = p4x_cm * beta_x + p4y_cm * beta_y + p4z_cm * beta_z;
    
    p4[0] = gamma * (E4_cm + p4_dot_beta);
    p4[1] = p4x_cm + boost_factor * p4_dot_beta * beta_x + gamma * beta_x * E4_cm;
    p4[2] = p4y_cm + boost_factor * p4_dot_beta * beta_y + gamma * beta_y * E4_cm;
    p4[3] = p4z_cm + boost_factor * p4_dot_beta * beta_z + gamma * beta_z * E4_cm;
}

}  // namespace smash

