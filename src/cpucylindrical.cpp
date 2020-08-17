#include <complex>

//real radial kinetic term
void block_kernel_radial_kinetic(size_t start_offset, size_t stride, size_t width, size_t height,
                                 double offset_x, double _kin_radial,
                                 double * p_real, double * p_imag) {

    double tmp_r, tmp_i;
    int start_i = 0;

    // The first two points of the radial coordinate have a different coupling
    if (offset_x + start_offset == 0) {
        double kin_radial = 2 * _kin_radial;
        double a = cos(kin_radial), b = - sin(kin_radial);
        for (size_t j = 0, idx = 0, peer = idx + 1; j < height; j += 1, idx += stride, peer += stride) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * tmp_r - b * p_imag[peer];
            p_imag[idx] = a * tmp_i + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_i;
            p_imag[peer] = a * p_imag[peer] + b * tmp_r;
        }
        start_i = 2;
    }

    double x = double(start_i) + offset_x + start_offset;
    for (int i = start_i + start_offset; i < width - 1; i += 2, x += 2) {
        double kin_radial = _kin_radial / sqrt(x * x - 0.25);
        double ratio = sqrt((2 * x + 1) / (2 * x - 1));
        double sinh_kin_radial = sinh(kin_radial);
        double a = cosh(kin_radial), b = sinh_kin_radial * ratio, c = - sinh_kin_radial / ratio;
        for (size_t j = 0, idx = i, peer = idx + 1; j < height; j += 1, idx += stride, peer += stride) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * tmp_r - b * p_imag[peer];
            p_imag[idx] = a * tmp_i + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - c * tmp_i;
            p_imag[peer] = a * p_imag[peer] + c * tmp_r;
        }
    }


}

//imaginary radial kinetic term
void block_kernel_radial_kinetic_imaginary(size_t start_offset, size_t stride, size_t width, size_t height,
        double offset_x, double _kin_radial,
        double * p_real, double * p_imag) {

    double tmp_r, tmp_i;
    int start_i = 0;

    // The first two points of the radial coordinate have a different coupling
    if (offset_x + start_offset == 0) {
        double kin_radial = 2 * _kin_radial;
        double a = cosh(kin_radial), b = -sinh(kin_radial);
        for (size_t j = 0, idx = 0, peer = idx + 1; j < height; j += 1, idx += stride, peer += stride) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * tmp_r + b * p_real[peer];
            p_imag[idx] = a * tmp_i + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] + b * tmp_r;
            p_imag[peer] = a * p_imag[peer] + b * tmp_i;
        }
        start_i = 2;
    }

    double x = double(start_i) + offset_x + start_offset;
    for (int i = start_i + start_offset; i < width - 1; i += 2, x += 2) {
        double kin_radial = _kin_radial / sqrt(x * x - 0.25);
        double ratio = sqrt((2 * x + 1) / (2 * x - 1));
        double sin_kin_radial = sin(kin_radial);
        double a = cos(kin_radial), b = sin_kin_radial * ratio, c = - sin_kin_radial / ratio;
        for (size_t j = 0, idx = i, peer = idx + 1; j < height; j += 1, idx += stride, peer += stride) {
            tmp_r = p_real[idx], tmp_i = p_imag[idx];
            p_real[idx] = a * tmp_r + b * p_real[peer];
            p_imag[idx] = a * tmp_i + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] + c * tmp_r;
            p_imag[peer] = a * p_imag[peer] + c * tmp_i;
        }
    }


}
