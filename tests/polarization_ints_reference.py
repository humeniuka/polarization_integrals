#!/usr/bin/env python
"""
reference implementation of polarization integrals by Liu Xiao
"""
import numpy as np
from numpy import sqrt, exp, pi
from scipy.special import gamma, gammainc, comb, dawsn, erf
from functools import reduce
import operator

def factor_double(n):
    if n<0 or n==0 or type(n) != int : return 1
    if n==1: return 1
    else: return reduce(operator.mul,range(n,0,-2))

def C_func(i, p):
    C_func_value = 1
    for v in range(1, i+1):
        C_func_value *= pow(p+0.5-v, -1)
    return C_func_value
    
def d_func(d_first_term, d_second_term, d_third_term):
    # Reference: eq 22 in Schwerdtfeger's paper
    d_value = 0

    if d_first_term > 0:
        p = int(d_first_term - 0.5)
        d_pos_sum_loop_value = 0
        for i in range(p):
            d_pos_sum_loop_value += ( pow(-1,i) * pow(d_second_term, i+0.5) * pow(2,i) / factor_double(2*i+1) )
        d_value = factor_double(2*p-1) * pow(2, 1-p) * pow(-1,p) * exp(d_second_term + d_third_term) * (dawsn(sqrt(d_second_term)) - d_pos_sum_loop_value)

    if d_first_term < 0:
        p = int(0.5 - d_first_term)
        d_neg_sum_loop_value = 0
        for i in range(1, p+1):
            d_neg_sum_loop_value += ( C_func(i, p) * pow(d_second_term, i-p-0.5) )
        d_value = exp(d_second_term + d_third_term) * (2 * C_func(p,p) * dawsn((sqrt(d_second_term))) - d_neg_sum_loop_value)

    return d_value

def m_func(m_para):
    m_value = exp(m_para*m_para) * erf(m_para) * dawsn(m_para)
    return m_value

def h_func(h_first_term, h_second_term):
    if h_first_term == 0:
        #h_value = pow(h_second_term, -0.5) * erf(sqrt(h_second_term))
        # Because the error function in Schwerdtfeger's paper differs from the usual definition
        # we have to add a factor or sqrt(pi)/2
        h_value = sqrt(pi)/2.0 * pow(h_second_term, -0.5) * erf(sqrt(h_second_term))
        return h_value
    elif h_first_term == 1:
        #h_value = 2 * exp(-h_second_term) * m_func(sqrt(h_second_term))
        h_value = 1.0/sqrt(pi) * exp(-h_second_term) * m_func(sqrt(h_second_term))
        return h_value
    else:
        h_value = (2 * (h_first_term + h_second_term) - 3)/2/(h_first_term-1) * h_func(h_first_term-1, h_second_term) - h_second_term/(h_first_term-1) * h_func(h_first_term-2, h_second_term)
        return h_value


# Part of Xiao's code for a single pair of unnormalized primitive Gaussians
def polarization_integral(x_1, y_1, z_1,  x_exp_i, y_exp_i, z_exp_i, alpha_ii,
                          x_2, y_2, z_2,  x_exp_j, y_exp_j, z_exp_j, alpha_jj,
                          pol_power_r,  pol_power_x, pol_power_y, pol_power_z,
                          cutoff_alpha, cutoff_power):
    pol_int_value = 0.0
    
    r_1_sq = pow(x_1, 2) + pow(y_1, 2) + pow(z_1, 2)
    r_1 = sqrt(r_1_sq)
    r_2_sq = pow(x_2, 2) + pow(y_2, 2) + pow(z_2, 2)
    r_2 = sqrt(r_2_sq)
    
    b_x1, b_y1, b_z1 = alpha_ii * x_1, alpha_ii * y_1, alpha_ii * z_1
    b_x2, b_y2, b_z2 = alpha_jj * x_2, alpha_jj * y_2, alpha_jj * z_2

    b_x, b_y, b_z = b_x1 + b_x2, b_y1 + b_y2, b_z1 + b_z2
    b_sq = pow(b_x, 2)+ pow(b_y, 2) + pow(b_z, 2)
    b = sqrt(b_sq)
    d_third_term = -(alpha_ii * r_1_sq + alpha_jj * r_2_sq)

    const_k = 1/gamma(pol_power_r/2) * pow(np.pi, 1.5)
    
    for x_exp_ii in range(x_exp_i + 1):
        const_x1 = comb(x_exp_i, x_exp_ii) * pow(-x_1, x_exp_i - x_exp_ii)
        coeff_k_nf_x1 = const_k * const_x1
        for x_exp_jj in range(x_exp_j + 1):
            const_x2 = comb(x_exp_j, x_exp_jj) * pow(-x_2, x_exp_j - x_exp_jj)
            coeff_k_nf_x1x2 = coeff_k_nf_x1 * const_x2
            x_pow = x_exp_ii + x_exp_jj + pol_power_x

            for y_exp_ii in range(y_exp_i + 1):
                const_y1 = comb(y_exp_i, y_exp_ii) * pow(-y_1, y_exp_i - y_exp_ii)
                coeff_k_nf_x1x2y1 = coeff_k_nf_x1x2 * const_y1
                for y_exp_jj in range(y_exp_j + 1):
                    const_y2 = comb(y_exp_j, y_exp_jj) * pow(-y_2, y_exp_j - y_exp_jj)
                    coeff_k_nf_x1x2y1y2 = coeff_k_nf_x1x2y1 * const_y2
                    y_pow = y_exp_ii + y_exp_jj + pol_power_y

                    for z_exp_ii in range(z_exp_i + 1):
                        const_z1 = comb(z_exp_i, z_exp_ii) * pow(-z_1, z_exp_i - z_exp_ii)
                        coeff_k_nf_x1x2y1y2z1 = coeff_k_nf_x1x2y1y2 * const_z1
                        for z_exp_jj in range(z_exp_j + 1):
                            const_z2 = comb(z_exp_j, z_exp_jj) * pow(-z_2, z_exp_j - z_exp_jj)
                            coeff_k_nf_x1x2y1y2z1z2 = coeff_k_nf_x1x2y1y2z1 * const_z2
                            z_pow = z_exp_ii + z_exp_jj + pol_power_z

                            for r_exp in range(cutoff_power + 1):
                                const_r = comb(cutoff_power, r_exp) * pow(-1, r_exp)
                                coeff_k_nf_x1x2y1y2z1z2_r = coeff_k_nf_x1x2y1y2z1z2 * const_r
                                e_power = cutoff_alpha * r_exp
                                a_u = alpha_ii + alpha_jj + e_power

                                for gx_exp in range(x_pow + 1):
                                    if (gx_exp % 2) != 0:
                                        pass
                                    else:
                                        const_gx = comb(x_pow, gx_exp) * factor_double(gx_exp-1) * pow(2, -gx_exp/2) * pow(b_x, x_pow - gx_exp)
                                        coeff_k_nf_x1x2y1y2z1z2_r_gx = coeff_k_nf_x1x2y1y2z1z2_r * const_gx

                                        for gy_exp in range(y_pow + 1):
                                            if (gy_exp % 2) != 0:
                                                pass
                                            else:
                                                const_gy = comb(y_pow, gy_exp) * factor_double(gy_exp-1) * pow(2, -gy_exp/2) * pow(b_y, y_pow - gy_exp)
                                                coeff_k_nf_x1x2y1y2z1z2_r_gxy = coeff_k_nf_x1x2y1y2z1z2_r_gx * const_gy
                                                
                                                for gz_exp in range(z_pow + 1):
                                                    if (gz_exp % 2) != 0:
                                                        pass
                                                    else:
                                                        const_gz = comb(z_pow, gz_exp) * factor_double(gz_exp-1) * pow(2, -gz_exp/2) * pow(b_z, z_pow - gz_exp)
                                                        coeff_k_nf_x1x2y1y2z1z2_r_gxyz = coeff_k_nf_x1x2y1y2z1z2_r_gxy * const_gz

                                                        s = x_pow + y_pow + z_pow - 0.5 * (gx_exp + gy_exp + gz_exp)

                                                        # Trick: combine the two exponential term
                                                        # Case 1: eq. 22 in Schwerdtfeger's paper, pol_power_r_half = j
                                                        if (pol_power_r % 2) == 0:
                                                            pol_power_r_half = int(pol_power_r / 2)
                                                            const_b = pow(b, -2 * s + pol_power_r - 3)
                                                            coeff_k_nf_x1x2y1y2z1z2_r_gxyz_b = coeff_k_nf_x1x2y1y2z1z2_r_gxyz * const_b

                                                            for v in range(pol_power_r_half):
                                                                d_first_term = s - pol_power_r_half + v + 1.5
                                                                d_second_term = b_sq / a_u
                                                                temp_value = coeff_k_nf_x1x2y1y2z1z2_r_gxyz_b * comb(pol_power_r_half-1, v) * pow(-a_u/b_sq, v) * d_func(d_first_term, d_second_term, d_third_term)
                                                                pol_int_value += temp_value
                                                                #print('Case1', x_pow, y_pow, z_pow, gx_exp, gy_exp, gz_exp, temp_value)

                                                        else:
                                                            pol_power_r_half = int((pol_power_r-1) / 2)
                                                            test_para = s-pol_power_r_half
                                                            
                                                            if (test_para).is_integer() == False:
                                                                pass
                                                                
                                                            else:
                                                                # Case 2, Subcase 1: eq. 32 in Schwerdtfeger's paper, pol_power_r_half = j
                                                                if (test_para) >= 0:
                                                                    const_a = pow(a_u, pol_power_r_half - s - 1) * pow(a_u/b_sq, pol_power_r_half+0.5) * exp(b_sq/a_u + d_third_term)
                                                                    coeff_k_nf_x1x2y1y2z1z2_r_gxyz_a = coeff_k_nf_x1x2y1y2z1z2_r_gxyz * const_a
                                                                    for v in range(int(s-pol_power_r_half)+1):
                                                                        g_first_term = pol_power_r_half + v + 0.5
                                                                        g_second_term = b_sq/a_u
                                                                        temp_value = coeff_k_nf_x1x2y1y2z1z2_r_gxyz_a * comb(int(s-pol_power_r_half), v) * pow(-a_u/b_sq, v) * gammainc(g_first_term, g_second_term) * gamma(g_first_term)
                                                                        pol_int_value += temp_value
                                                                        print('Case2a', x_pow, y_pow, z_pow, gx_exp, gy_exp, gz_exp)

                                                                # Case 2, Subcase 2: eq. 39 in Schwerdtfeger's paper
                                                                # Basically did not get involved in QM-He/MM-He system
                                                                else:
                                                                    const_a = 2 * pow(a_u, pol_power_r_half - s - 1) * exp(b_sq/a_u + d_third_term)
                                                                    coeff_k_nf_x1x2y1y2z1z2_r_gxyz_a = coeff_k_nf_x1x2y1y2z1z2_r_gxyz * const_a
                                                                    for v in range(int(pol_power_r_half) + 1):
                                                                        h_first_term = pol_power_r_half - s - v
                                                                        h_second_term = b_sq/a_u
                                                                        
                                                                        temp_value = coeff_k_nf_x1x2y1y2z1z2_r_gxyz_a * comb(int(pol_power_r_half), v) * pow(-1, v) * h_func(h_first_term, h_second_term)
                                                                        pol_int_value += temp_value
                                                                        print('Case2b', x_pow, y_pow, z_pow, gx_exp, gy_exp, gz_exp)

    return pol_int_value

def unique_integrals(x_1, y_1, z_1,  x_exp_i, y_exp_i, z_exp_i, alpha_ii,
                          x_2, y_2, z_2,  x_exp_j, y_exp_j, z_exp_j, alpha_jj,
                          pol_power_r,  pol_power_x, pol_power_y, pol_power_z,
                          cutoff_alpha, cutoff_power):
    
    r_1_sq = pow(x_1, 2) + pow(y_1, 2) + pow(z_1, 2)
    r_1 = sqrt(r_1_sq)
    r_2_sq = pow(x_2, 2) + pow(y_2, 2) + pow(z_2, 2)
    r_2 = sqrt(r_2_sq)
    
    b_x1, b_y1, b_z1 = alpha_ii * x_1, alpha_ii * y_1, alpha_ii * z_1
    b_x2, b_y2, b_z2 = alpha_jj * x_2, alpha_jj * y_2, alpha_jj * z_2

    b_x, b_y, b_z = b_x1 + b_x2, b_y1 + b_y2, b_z1 + b_z2
    b_sq = pow(b_x, 2)+ pow(b_y, 2) + pow(b_z, 2)
    b = sqrt(b_sq)
    d_third_term = -(alpha_ii * r_1_sq + alpha_jj * r_2_sq)

    const_k = 1/gamma(pol_power_r/2) * pow(np.pi, 1.5)

    smax = x_exp_i + x_exp_j + y_exp_i + y_exp_j + z_exp_i + z_exp_j + pol_power_x + pol_power_y + pol_power_z
    integrals = np.zeros(smax+1)
    
    for s in range(0, smax+1):
        for r_exp in range(cutoff_power + 1):
            const_r = comb(cutoff_power, r_exp) * pow(-1, r_exp)
            e_power = cutoff_alpha * r_exp
            a_u = alpha_ii + alpha_jj + e_power

            if (pol_power_r % 2) == 0:
                pol_power_r_half = int(pol_power_r / 2)
                const_b = pow(b, -2 * s + pol_power_r - 3)
                coeff = const_k * const_r * const_b
                for v in range(pol_power_r_half):
                    d_first_term = s - pol_power_r_half + v + 1.5
                    d_second_term = b_sq / a_u
                    temp_value = coeff * comb(pol_power_r_half-1, v) * pow(-a_u/b_sq, v) * d_func(d_first_term, d_second_term, d_third_term)
                    integrals[s] += temp_value

    return integrals
