
#include <string>
#include <complex>


void block_kernel_vertical(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag) {
    for (size_t idx = start_offset, peer = idx + stride; idx < width; idx += 2, peer += 2) {
        double tmp_real = p_real[idx];
        double tmp_imag = p_imag[idx];
        p_real[idx] = a * tmp_real - b * p_imag[peer];
        p_imag[idx] = a * tmp_imag + b * p_real[peer];
        p_real[peer] = a * p_real[peer] - b * tmp_imag;
        p_imag[peer] = a * p_imag[peer] + b * tmp_real;
    }
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + stride; idx < y * stride + width; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real - b * p_imag[peer];
            p_imag[idx] = a * tmp_imag + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_imag;
            p_imag[peer] = a * p_imag[peer] + b * tmp_real;
        }
    }
}

void block_kernel_vertical_imaginary(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag) {
    for (size_t idx = start_offset, peer = idx + stride; idx < width; idx += 2, peer += 2) {
        double tmp_real = p_real[idx];
        double tmp_imag = p_imag[idx];
        p_real[idx] = a * tmp_real + b * p_real[peer];
        p_imag[idx] = a * tmp_imag + b * p_imag[peer];
        p_real[peer] = a * p_real[peer] + b * tmp_real;
        p_imag[peer] = a * p_imag[peer] + b * tmp_imag;
    }
    for (size_t y = 1; y < height - 1; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + stride; idx < y * stride + width; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real + b * p_real[peer];
            p_imag[idx] = a * tmp_imag + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] + b * tmp_real;
            p_imag[peer] = a * p_imag[peer] + b * tmp_imag;
        }
    }
}

void block_kernel_horizontal(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + 1; idx < y * stride + width - 1; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real - b * p_imag[peer];
            p_imag[idx] = a * tmp_imag + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_imag;
            p_imag[peer] = a * p_imag[peer] + b * tmp_real;
        }
    }
}

void block_kernel_horizontal_imaginary(size_t start_offset, size_t stride, size_t width, size_t height, double a, double b, double * p_real, double * p_imag) {
    for (size_t y = 0; y < height; ++y) {
        for (size_t idx = y * stride + (start_offset + y) % 2, peer = idx + 1; idx < y * stride + width - 1; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real + b * p_real[peer];
            p_imag[idx] = a * tmp_imag + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] + b * tmp_real;
            p_imag[peer] = a * p_imag[peer] + b * tmp_imag;
        }
    }
}

//double time potential
void block_kernel_potential(bool two_wavefunctions, size_t stride, size_t width, size_t height, double coupling_a, double coupling_b, double coupling_aa, size_t tile_width,
                            const double *external_pot_real, const double *external_pot_imag, const double *pb_real, const double *pb_imag, double * p_real, double * p_imag) {
    if(two_wavefunctions) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t idx = y * stride, idx_pot = y * tile_width; idx < y * stride + width; ++idx, ++idx_pot) {
                double norm_2 = p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx];
                //double norm_3 = norm_2 * sqrt(norm_2);
                double norm_2b = pb_real[idx_pot] * pb_real[idx_pot] + pb_imag[idx_pot] * pb_imag[idx_pot];
                double c_cos = cos(coupling_a * norm_2 + coupling_b * norm_2b/* + coupling_aa * norm_3*/);
                double c_sin = sin(coupling_a * norm_2 + coupling_b * norm_2b/* + coupling_aa * norm_3*/);
                double tmp = p_real[idx];
                p_real[idx] = external_pot_real[idx_pot] * tmp - external_pot_imag[idx_pot] * p_imag[idx];
                p_imag[idx] = external_pot_real[idx_pot] * p_imag[idx] + external_pot_imag[idx_pot] * tmp;

                tmp = p_real[idx];
                p_real[idx] = c_cos * tmp + c_sin * p_imag[idx];
                p_imag[idx] = c_cos * p_imag[idx] - c_sin * tmp;
            }
        }
    }
    else {
        for (size_t y = 0; y < height; ++y) {
            for (size_t idx = y * stride, idx_pot = y * tile_width; idx < y * stride + width; ++idx, ++idx_pot) {
                double norm_2 = p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx];
                double norm_3 = norm_2 * sqrt(norm_2);
                double c_cos = cos(coupling_a * norm_2 + coupling_aa * norm_3);
                double c_sin = sin(coupling_a * norm_2 + coupling_aa * norm_3);
                double tmp = p_real[idx];
                p_real[idx] = external_pot_real[idx_pot] * tmp - external_pot_imag[idx_pot] * p_imag[idx];
                p_imag[idx] = external_pot_real[idx_pot] * p_imag[idx] + external_pot_imag[idx_pot] * tmp;

                tmp = p_real[idx];
                p_real[idx] = c_cos * tmp + c_sin * p_imag[idx];
                p_imag[idx] = c_cos * p_imag[idx] - c_sin * tmp;
            }
        }
    }
}

//double time potential
void block_kernel_potential_imaginary(bool two_wavefunctions, size_t stride, size_t width, size_t height, double coupling_a, double coupling_b, double coupling_aa, size_t tile_width,
                                      const double *external_pot_real, const double *external_pot_imag, const double *pb_real, const double *pb_imag, double * p_real, double * p_imag) {
    if(two_wavefunctions) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t idx = y * stride, idx_pot = y * tile_width; idx < y * stride + width; ++idx, ++idx_pot) {
                double norm_2 = p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx];
                //double norm_3 = norm_2 * sqrt(norm_2);
                double norm_2b = pb_real[idx_pot] * pb_real[idx_pot] + pb_imag[idx_pot] * pb_imag[idx_pot];
                double tmp = exp(-1. * (coupling_a * norm_2 + coupling_b * norm_2b/* + coupling_aa * norm_3*/));
                p_real[idx] = tmp * external_pot_real[idx_pot] * p_real[idx];
                p_imag[idx] = tmp * external_pot_real[idx_pot] * p_imag[idx];
            }
        }
    }
    else {
        for (size_t y = 0; y < height; ++y) {
            for (size_t idx = y * stride, idx_pot = y * tile_width; idx < y * stride + width; ++idx, ++idx_pot) {
                double norm_2 = p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx];
                double norm_3 = norm_2 * sqrt(norm_2);
                double tmp = exp(-1. * (coupling_a * norm_2 + coupling_aa * norm_3));
                p_real[idx] = tmp * external_pot_real[idx_pot] * p_real[idx];
                p_imag[idx] = tmp * external_pot_real[idx_pot] * p_imag[idx];
            }
        }
    }
}

//rotation
void block_kernel_rotation(size_t stride, size_t width, size_t height, int offset_x, int offset_y, double alpha_x, double alpha_y, double * p_real, double * p_imag) {

    double tmp_r, tmp_i;

    for (int j = 0, y = offset_y; j < height; ++j, ++y) {
        double alpha_yy = - 0.5 * alpha_y * y;
        double a = cos(alpha_yy), b = sin(alpha_yy);
        for (size_t i = 0, idx = j * stride, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_real[peer];
            p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_r;
            p_imag[peer] = a * p_imag[peer] - b * tmp_i;
        }
        for (size_t i = 1, idx = j * stride + 1, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_real[peer];
            p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_r;
            p_imag[peer] = a * p_imag[peer] - b * tmp_i;
        }
    }

    for (int i = 0, x = offset_x; i < width; ++i, ++x) {
        double alpha_xx = alpha_x * x;
        double a = cos(alpha_xx), b = sin(alpha_xx);
        for (size_t j = 0, idx = i, peer = stride + idx; j < height - 1; j += 2, idx += 2 * stride, peer += 2 * stride) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_real[peer];
            p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_r;
            p_imag[peer] = a * p_imag[peer] - b * tmp_i;
        }
        for (size_t j = 1, idx = j * stride + i, peer = stride + idx; j < height - 1; j += 2, idx += 2 * stride, peer += 2 * stride) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_real[peer];
            p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_r;
            p_imag[peer] = a * p_imag[peer] - b * tmp_i;
        }
    }

    for (int j = 0, y = offset_y; j < height; ++j, ++y) {
        double alpha_yy = - 0.5 * alpha_y * y;
        double a = cos(alpha_yy), b = sin(alpha_yy);
        for (size_t i = 0, idx = j * stride, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_real[peer];
            p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_r;
            p_imag[peer] = a * p_imag[peer] - b * tmp_i;
        }
        for (size_t i = 1, idx = j * stride + 1, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_real[peer];
            p_imag[idx] = a * p_imag[idx] + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_r;
            p_imag[peer] = a * p_imag[peer] - b * tmp_i;
        }
    }
}

void block_kernel_rotation_imaginary(size_t stride, size_t width, size_t height, int offset_x, int offset_y, double alpha_x, double alpha_y, double * p_real, double * p_imag) {

    double tmp_r, tmp_i;
    for (int j = 0, y = offset_y; j < height; ++j, ++y) {
        double alpha_yy = - 0.5 * alpha_y * y;
        double a = cosh(alpha_yy), b = sinh(alpha_yy);
        for (size_t i = 0, idx = j * stride, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_imag[peer];
            p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
            p_real[peer] = - b * tmp_i + a * p_real[peer];
            p_imag[peer] = b * tmp_r + a * p_imag[peer];
        }
        for (size_t i = 1, idx = j * stride + 1, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_imag[peer];
            p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
            p_real[peer] = - b * tmp_i + a * p_real[peer];
            p_imag[peer] = b * tmp_r + a * p_imag[peer];
        }
    }

    for (int i = 0, x = offset_x; i < width; ++i, ++x) {
        double alpha_xx = alpha_x * x;
        double a = cosh(alpha_xx), b = sinh(alpha_xx);
        for (size_t j = 0, idx = i, peer = stride + idx; j < height - 1; j += 2, idx += 2 * stride, peer += 2 * stride) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_imag[peer];
            p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
            p_real[peer] = - b * tmp_i + a * p_real[peer];
            p_imag[peer] = b * tmp_r + a * p_imag[peer];
        }
        for (size_t j = 1, idx = j * stride + i, peer = stride + idx; j < height - 1; j += 2, idx += 2 * stride, peer += 2 * stride) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_imag[peer];
            p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
            p_real[peer] = - b * tmp_i + a * p_real[peer];
            p_imag[peer] = b * tmp_r + a * p_imag[peer];
        }
    }

    for (int j = 0, y = offset_y; j < height; ++j, ++y) {
        double alpha_yy = - 0.5 * alpha_y * y;
        double a = cosh(alpha_yy), b = sinh(alpha_yy);
        for (size_t i = 0, idx = j * stride, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_imag[peer];
            p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
            p_real[peer] = - b * tmp_i + a * p_real[peer];
            p_imag[peer] = b * tmp_r + a * p_imag[peer];
        }
        for (size_t i = 1, idx = j * stride + 1, peer = idx + 1; i < width - 1; i += 2, idx += 2, peer += 2) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * p_real[idx] + b * p_imag[peer];
            p_imag[idx] = a * p_imag[idx] - b * p_real[peer];
            p_real[peer] = - b * tmp_i + a * p_real[peer];
            p_imag[peer] = b * tmp_r + a * p_imag[peer];
        }
    }
}

void rabi_coupling_real(size_t stride, size_t width, size_t height, double cc, double cs_r, double cs_i, double *p_real, double *p_imag, double *pb_real, double *pb_imag) {
    double real, imag;
    for(size_t i = 0; i < height; i++) {
        for(size_t j = 0, idx = i * stride; j < width; j++, idx++) {
            real = p_real[idx];
            imag = p_imag[idx];
            p_real[idx] = cc * real - cs_i * pb_real[idx] - cs_r * pb_imag[idx];
            p_imag[idx] = cc * imag + cs_r * pb_real[idx] - cs_i * pb_imag[idx];
            pb_real[idx] = cc * pb_real[idx] + cs_i * real - cs_r * imag;
            pb_imag[idx] = cc * pb_imag[idx] + cs_r * real + cs_i * imag;
        }
    }
}

void rabi_coupling_imaginary(size_t stride, size_t width, size_t height, double cc, double cs_r, double cs_i, double *p_real, double *p_imag, double *pb_real, double *pb_imag) {
    double real, imag;
    for(size_t i = 0; i < height; i++) {
        for(size_t j = 0, idx = i * stride; j < width; j++, idx++) {
            real = p_real[idx];
            imag = p_imag[idx];
            p_real[idx] = cc * real + cs_r * pb_real[idx] - cs_i * pb_imag[idx];
            p_imag[idx] = cc * imag + cs_i * pb_real[idx] + cs_r * pb_imag[idx];
            pb_real[idx] = cc * pb_real[idx] + cs_r * real + cs_i * imag;
            pb_imag[idx] = cc * pb_imag[idx] - cs_i * real + cs_r * imag;
        }
    }
}
