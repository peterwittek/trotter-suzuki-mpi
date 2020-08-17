/**
 * Massively Parallel Trotter-Suzuki Solver
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
//#include "pch.h"
#include "common.h"
#include "kernel.h"
#include <iostream>


void full_step(bool two_wavefunctions, size_t stride, size_t width, size_t height,
               double offset_x, double offset_y, double alpha_x, double alpha_y,
               double aH, double bH, double aV, double bV, double kin_radial, double coupling_a, double coupling_b, double coupling_aa,
               size_t tile_width, const double *external_pot_real, const double *external_pot_imag,
               const double *pb_real, const double *pb_imag, double * real, double * imag,
               string coordinate_system) {
    if (height > 1 ) {
        block_kernel_vertical  (0u, stride, width, height, aV, bV, real, imag);
    }
    block_kernel_horizontal(0u, stride, width, height, aH, bH, real, imag);
    if (height > 1 ) {
        block_kernel_vertical  (1u, stride, width, height, aV, bV, real, imag);
    }
    block_kernel_horizontal(1u, stride, width, height, aH, bH, real, imag);
    if (coordinate_system == "cylindrical") {
        block_kernel_radial_kinetic(0u, stride, width, height, offset_x, kin_radial, real, imag);
        block_kernel_radial_kinetic(1u, stride, width, height, offset_x, kin_radial, real, imag);
    }
    block_kernel_potential (two_wavefunctions, stride, width, height, coupling_a, coupling_b, coupling_aa, tile_width, external_pot_real, external_pot_imag, pb_real, pb_imag, real, imag);
    if (alpha_x != 0. && alpha_y != 0.) {
        block_kernel_rotation  (stride, width, height, offset_x, offset_y, alpha_x, alpha_y, real, imag);
    }
    if (coordinate_system == "cylindrical") {
        block_kernel_radial_kinetic(1u, stride, width, height, offset_x, kin_radial, real, imag);
        block_kernel_radial_kinetic(0u, stride, width, height, offset_x, kin_radial, real, imag);
    }
    block_kernel_horizontal(1u, stride, width, height, aH, bH, real, imag);
    if (height > 1 ) {
        block_kernel_vertical  (1u, stride, width, height, aV, bV, real, imag);
    }
    block_kernel_horizontal(0u, stride, width, height, aH, bH, real, imag);
    if (height > 1 ) {
        block_kernel_vertical  (0u, stride, width, height, aV, bV, real, imag);
    }
}

void full_step_imaginary(bool two_wavefunctions, size_t stride, size_t width, size_t height,
                         double offset_x, double offset_y, double alpha_x, double alpha_y,
                         double aH, double bH, double aV, double bV, double kin_radial, double coupling_a, double coupling_b, double coupling_aa,
                         size_t tile_width, const double *external_pot_real, const double *external_pot_imag,
                         const double *pb_real, const double *pb_imag, double * real, double * imag,
                         string coordinate_system) {
    if (height > 1 ) {
        block_kernel_vertical_imaginary  (0u, stride, width, height, aV, bV, real, imag);
    }
    block_kernel_horizontal_imaginary(0u, stride, width, height, aH, bH, real, imag);
    if (height > 1 ) {
        block_kernel_vertical_imaginary  (1u, stride, width, height, aV, bV, real, imag);
    }
    block_kernel_horizontal_imaginary(1u, stride, width, height, aH, bH, real, imag);
    if (coordinate_system == "cylindrical") {
        block_kernel_radial_kinetic_imaginary(0u, stride, width, height, offset_x, kin_radial, real, imag);
        block_kernel_radial_kinetic_imaginary(1u, stride, width, height, offset_x, kin_radial, real, imag);
    }
    block_kernel_potential_imaginary (two_wavefunctions, stride, width, height, coupling_a, coupling_b, coupling_aa, tile_width, external_pot_real, external_pot_imag, pb_real, pb_imag, real, imag);
    if (alpha_x != 0. && alpha_y != 0.) {
        block_kernel_rotation_imaginary(stride, width, height, offset_x, offset_y, alpha_x, alpha_y, real, imag);
    }
    if (coordinate_system == "cylindrical") {
        block_kernel_radial_kinetic_imaginary(1u, stride, width, height, offset_x, kin_radial, real, imag);
        block_kernel_radial_kinetic_imaginary(0u, stride, width, height, offset_x, kin_radial, real, imag);
    }
    block_kernel_horizontal_imaginary(1u, stride, width, height, aH, bH, real, imag);
    if (height > 1 ) {
        block_kernel_vertical_imaginary  (1u, stride, width, height, aV, bV, real, imag);
    }
    block_kernel_horizontal_imaginary(0u, stride, width, height, aH, bH, real, imag);
    if (height > 1 ) {
        block_kernel_vertical_imaginary  (0u, stride, width, height, aV, bV, real, imag);
    }
}

void process_sides(bool two_wavefunctions, double offset_tile_x, double offset_tile_y, double alpha_x, double alpha_y, size_t tile_width, size_t block_width, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height,
                   double aH, double bH, double aV, double bV, double kin_radial, double coupling_a, double coupling_b, double coupling_aa, const double *external_pot_real, const double *external_pot_imag,
                   const double * p_real, const double * p_imag, const double * pb_real, const double * pb_imag,
                   double * next_real, double * next_imag, double * block_real, double * block_imag, bool imag_time, string coordinate_system) {

    // First block [0..block_width - halo_x]
    memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width], tile_width * sizeof(double), block_width * sizeof(double), read_height);
    memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width], tile_width * sizeof(double), block_width * sizeof(double), read_height);
    if(imag_time)
        full_step_imaginary(two_wavefunctions, block_width, block_width, read_height, offset_tile_x, offset_tile_y + read_y, alpha_x, alpha_y, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, tile_width,
                            &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], &pb_real[read_y * tile_width], &pb_imag[read_y * tile_width], block_real, block_imag, coordinate_system);
    else
        full_step(two_wavefunctions, block_width, block_width, read_height, offset_tile_x, offset_tile_y + read_y, alpha_x, alpha_y, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, tile_width,
                  &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], &pb_real[read_y * tile_width], &pb_imag[read_y * tile_width], block_real, block_imag, coordinate_system);
    memcpy2D(&next_real[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_real[write_offset * block_width], block_width * sizeof(double), (block_width - halo_x) * sizeof(double), write_height);
    memcpy2D(&next_imag[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_imag[write_offset * block_width], block_width * sizeof(double), (block_width - halo_x) * sizeof(double), write_height);

    size_t block_start = ((tile_width - block_width) / (block_width - 2 * halo_x) + 1) * (block_width - 2 * halo_x);
    // Last block
    memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width + block_start], tile_width * sizeof(double), (tile_width - block_start) * sizeof(double), read_height);
    memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width + block_start], tile_width * sizeof(double), (tile_width - block_start) * sizeof(double), read_height);
    if(imag_time)
        full_step_imaginary(two_wavefunctions, block_width, tile_width - block_start, read_height, offset_tile_x + block_start, offset_tile_y + read_y, alpha_x, alpha_y, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, tile_width,
                            &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], &pb_real[read_y * tile_width + block_start], &pb_imag[read_y * tile_width + block_start], block_real, block_imag, coordinate_system);
    else
        full_step(two_wavefunctions, block_width, tile_width - block_start, read_height, offset_tile_x + block_start, offset_tile_y + read_y, alpha_x, alpha_y, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, tile_width,
                  &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], &pb_real[read_y * tile_width + block_start], &pb_imag[read_y * tile_width + block_start], block_real, block_imag, coordinate_system);
    memcpy2D(&next_real[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_real[write_offset * block_width + halo_x], block_width * sizeof(double), (tile_width - block_start - halo_x) * sizeof(double), write_height);
    memcpy2D(&next_imag[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_imag[write_offset * block_width + halo_x], block_width * sizeof(double), (tile_width - block_start - halo_x) * sizeof(double), write_height);
}

void process_band(bool two_wavefunctions, double offset_tile_x, double offset_tile_y, double alpha_x, double alpha_y, size_t tile_width, size_t block_width, size_t block_height, size_t halo_x, size_t read_y, size_t read_height, size_t write_offset, size_t write_height,
                  double aH, double bH, double aV, double bV, double kin_radial, double coupling_a, double coupling_b, double coupling_aa, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag,
                  const double * pb_real, const double * pb_imag, double * next_real, double * next_imag, int inner, int sides, bool imag_time, string coordinate_system) {
    double *block_real = new double[block_height * block_width];
    double *block_imag = new double[block_height * block_width];

    if (tile_width <= block_width) {
        if (sides) {
            // One full block
            memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width], tile_width * sizeof(double), tile_width * sizeof(double), read_height);
            memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width], tile_width * sizeof(double), tile_width * sizeof(double), read_height);
            if(imag_time)
                full_step_imaginary(two_wavefunctions, block_width, tile_width, read_height, offset_tile_x, offset_tile_y + read_y, alpha_x, alpha_y, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, tile_width,
                                    &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], &pb_real[read_y * tile_width], &pb_imag[read_y * tile_width], block_real, block_imag, coordinate_system);
            else
                full_step(two_wavefunctions, block_width, tile_width, read_height, offset_tile_x, offset_tile_y + read_y, alpha_x, alpha_y, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, tile_width,
                          &external_pot_real[read_y * tile_width], &external_pot_imag[read_y * tile_width], &pb_real[read_y * tile_width], &pb_imag[read_y * tile_width], block_real, block_imag, coordinate_system);
            memcpy2D(&next_real[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_real[write_offset * block_width], block_width * sizeof(double), tile_width * sizeof(double), write_height);
            memcpy2D(&next_imag[(read_y + write_offset) * tile_width], tile_width * sizeof(double), &block_imag[write_offset * block_width], block_width * sizeof(double), tile_width * sizeof(double), write_height);
        }
    }
    else {
        if (sides) {
            process_sides(two_wavefunctions, offset_tile_x, offset_tile_y, alpha_x, alpha_y, tile_width, block_width, halo_x, read_y, read_height, write_offset, write_height, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, external_pot_real, external_pot_imag, p_real, p_imag, pb_real, pb_imag, next_real, next_imag, block_real, block_imag, imag_time, coordinate_system);
        }
        if (inner) {
            for (size_t block_start = block_width - 2 * halo_x; block_start < tile_width - block_width; block_start += block_width - 2 * halo_x) {
                memcpy2D(block_real, block_width * sizeof(double), &p_real[read_y * tile_width + block_start], tile_width * sizeof(double), block_width * sizeof(double), read_height);
                memcpy2D(block_imag, block_width * sizeof(double), &p_imag[read_y * tile_width + block_start], tile_width * sizeof(double), block_width * sizeof(double), read_height);
                if(imag_time)
                    full_step_imaginary(two_wavefunctions, block_width, block_width, read_height, offset_tile_x + block_start, offset_tile_y + read_y, alpha_x, alpha_y, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, tile_width,
                                        &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], &pb_real[read_y * tile_width + block_start], &pb_imag[read_y * tile_width + block_start], block_real, block_imag, coordinate_system);
                else
                    full_step(two_wavefunctions, block_width, block_width, read_height, offset_tile_x + block_start, offset_tile_y + read_y, alpha_x, alpha_y, aH, bH, aV, bV, kin_radial, coupling_a, coupling_b, coupling_aa, tile_width,
                              &external_pot_real[read_y * tile_width + block_start], &external_pot_imag[read_y * tile_width + block_start], &pb_real[read_y * tile_width + block_start], &pb_imag[read_y * tile_width + block_start], block_real, block_imag, coordinate_system);
                memcpy2D(&next_real[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_real[write_offset * block_width + halo_x], block_width * sizeof(double), (block_width - 2 * halo_x) * sizeof(double), write_height);
                memcpy2D(&next_imag[(read_y + write_offset) * tile_width + block_start + halo_x], tile_width * sizeof(double), &block_imag[write_offset * block_width + halo_x], block_width * sizeof(double), (block_width - 2 * halo_x) * sizeof(double), write_height);
            }
        }
    }


    delete[] block_real;
    delete[] block_imag;
}

// Class methods
CPUBlock::CPUBlock(Lattice *grid, State *state, Hamiltonian *hamiltonian,
                   double *_external_pot_real, double *_external_pot_imag,
                   double delta_t, double _norm, bool _imag_time):
    sense(0),
    state_index(0),
    imag_time(_imag_time) {
    delta_x = grid->delta_x;
    delta_y = grid->delta_y;
    halo_x = grid->halo_x;
    halo_y = grid->halo_y;
    periods = grid->periods;
    rot_coord_x = hamiltonian->rot_coord_x;
    rot_coord_y = hamiltonian->rot_coord_y;
    alpha_x = hamiltonian->angular_velocity * delta_t * grid->delta_x / (2 * grid->delta_y);
    alpha_y = hamiltonian->angular_velocity * delta_t * grid->delta_y / (2 * grid->delta_x);
    coupling_const = new double [3];
    LeeHuangYang_coupling = new double [2];
    norm = new double [1];
    aH = new double [1];
    bH = new double [1];
    aV = new double [1];
    bV = new double [1];
    kin_radial = new double [1];
    if (imag_time) {
        aH[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
        bH[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
        aV[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
        bV[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
        if (grid->coordinate_system == "cylindrical") {
            kin_radial[0] = delta_t / (8. * hamiltonian->mass * grid->delta_x * grid->delta_x);
        }
    }
    else {
        aH[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
        bH[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
        aV[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
        bV[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
        if (grid->coordinate_system == "cylindrical") {
            kin_radial[0] = delta_t / (8. * hamiltonian->mass * grid->delta_x * grid->delta_x);
        }
    }
    coupling_const[0] = hamiltonian->coupling_a * delta_t;
    coupling_const[1] = 0.;
    coupling_const[2] = 0.;
    LeeHuangYang_coupling[0] = hamiltonian->LeeHuangYang_coupling_a  * delta_t;
    LeeHuangYang_coupling[1] = 0;
    norm[0] = _norm;
    tot_norm = norm[0];
    coordinate_system = grid->coordinate_system;
    angular_momentum[0] = state->angular_momentum;
#ifdef HAVE_MPI
    cartcomm = grid->cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
#endif
    if (halo_y == 0) {
        block_height = 1;
    }
    else {
        block_height = BLOCK_HEIGHT_CACHE;
    }
    start_x = grid->start_x;
    end_x = grid->end_x;
    inner_start_x = grid->inner_start_x;
    inner_end_x = grid->inner_end_x;
    start_y = grid->start_y;
    end_y = grid->end_y;
    inner_start_y = grid->inner_start_y;
    inner_end_y = grid->inner_end_y;
    tile_width = end_x - start_x;
    tile_height = end_y - start_y;

    p_real[0][0] = state->p_real;
    p_imag[0][0] = state->p_imag;
    p_real[0][1] = new double[tile_width * tile_height];
    p_imag[0][1] = new double[tile_width * tile_height];
    p_real[1][0] = NULL;
    p_imag[1][0] = NULL;
    p_real[1][1] = NULL;
    p_imag[1][1] = NULL;
    external_pot_real[0] = _external_pot_real;
    external_pot_imag[0] = _external_pot_imag;
    two_wavefunctions = false;

#ifdef HAVE_MPI
    // Halo exchange uses wave pattern to communicate
    // halo_x-wide inner rows are sent first to left and right
    // Then full length rows are exchanged to the top and bottom
    int count = inner_end_y - inner_start_y;  // The number of rows in the halo submatrix
    int block_length = halo_x;  // The number of columns in the halo submatrix
    int stride = tile_width;  // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &verticalBorder);
    MPI_Type_commit (&verticalBorder);

    count = halo_y; // The vertical halo in rows
    block_length = tile_width;  // The number of columns of the matrix
    stride = tile_width;  // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &horizontalBorder);
    MPI_Type_commit (&horizontalBorder);
#endif
}

CPUBlock::CPUBlock(Lattice *grid, State *state1, State *state2,
                   Hamiltonian2Component *hamiltonian,
                   double **_external_pot_real, double **_external_pot_imag,
                   double delta_t, double *_norm, bool _imag_time):
    sense(0),
    state_index(0),
    imag_time(_imag_time) {
    delta_x = grid->delta_x;
    delta_y = grid->delta_y;
    halo_x = grid->halo_x;
    halo_y = grid->halo_y;
    alpha_x = hamiltonian->angular_velocity * delta_t * grid->delta_x / (2 * grid->delta_y),
    alpha_y = hamiltonian->angular_velocity * delta_t * grid->delta_y / (2 * grid->delta_x),
    rot_coord_x = hamiltonian->rot_coord_x;
    rot_coord_y = hamiltonian->rot_coord_y;
    aH = new double [2];
    bH = new double [2];
    aV = new double [2];
    bV = new double [2];
    kin_radial = new double [2];
    if (imag_time) {
        aH[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
        bH[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
        aH[1] = cosh(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_x));
        bH[1] = sinh(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_x));
        aV[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
        bV[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
        aV[1] = cosh(delta_t / (4. * hamiltonian->mass_b * grid->delta_y * grid->delta_y));
        bV[1] = sinh(delta_t / (4. * hamiltonian->mass_b * grid->delta_y * grid->delta_y));
        if (grid->coordinate_system == "cylindrical") {
            kin_radial[0] = delta_t / (8. * hamiltonian->mass * grid->delta_x * grid->delta_x);
            kin_radial[1] = delta_t / (8. * hamiltonian->mass_b * grid->delta_x * grid->delta_x);
        }
    }
    else {
        aH[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
        bH[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
        aH[1] = cos(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_x));
        bH[1] = sin(delta_t / (4. * hamiltonian->mass_b * grid->delta_x * grid->delta_x));
        aV[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
        bV[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_y * grid->delta_y));
        aV[1] = cos(delta_t / (4. * hamiltonian->mass_b * grid->delta_y * grid->delta_y));
        bV[1] = sin(delta_t / (4. * hamiltonian->mass_b * grid->delta_y * grid->delta_y));
        if (grid->coordinate_system == "cylindrical") {
            kin_radial[0] = delta_t / (8. * hamiltonian->mass * grid->delta_x * grid->delta_x);
            kin_radial[1] = delta_t / (8. * hamiltonian->mass_b * grid->delta_x * grid->delta_x);
        }
    }
    norm = new double [2];
    norm[0] = _norm[0];
    norm[1] = _norm[1];
    tot_norm = norm[0] + norm[1];
    coupling_const = new double[5];
    coupling_const[0] = delta_t * hamiltonian->coupling_a;
    coupling_const[1] = delta_t * hamiltonian->coupling_b;
    coupling_const[2] = delta_t * hamiltonian->coupling_ab;
    coupling_const[3] = 0.5 * hamiltonian->omega_r;
    coupling_const[4] = 0.5 * hamiltonian->omega_i;
    LeeHuangYang_coupling = new double [2];
    // LeeHuangYang_coupling[0] = hamiltonian->LeeHuangYang_coupling_a  * delta_t;
    // LeeHuangYang_coupling[1] = hamiltonian->LeeHuangYang_coupling_b  * delta_t;
    periods = grid->periods;
    coordinate_system = grid->coordinate_system;
    angular_momentum[0] = state1->angular_momentum;
    angular_momentum[1] = state2->angular_momentum;
#ifdef HAVE_MPI
    cartcomm = grid->cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
#endif
    if (halo_y == 0) {
        block_height = 1;
    }
    else {
        block_height = BLOCK_HEIGHT_CACHE;
    }

    start_x = grid->start_x;
    end_x = grid->end_x;
    inner_start_x = grid->inner_start_x;
    inner_end_x = grid->inner_end_x;
    start_y = grid->start_y;
    end_y = grid->end_y;
    inner_start_y = grid->inner_start_y;
    inner_end_y = grid->inner_end_y;
    tile_width = end_x - start_x;
    tile_height = end_y - start_y;
    p_real[0][0] = state1->p_real;
    p_imag[0][0] = state1->p_imag;
    p_real[1][0] = state2->p_real;
    p_imag[1][0] = state2->p_imag;

    for(int i = 0; i < 2; i++) {
        p_real[i][1] = new double[tile_width * tile_height];
        p_imag[i][1] = new double[tile_width * tile_height];
        memcpy2D(p_real[i][1], tile_width * sizeof(double), p_real[i][0], tile_width * sizeof(double), tile_width * sizeof(double), tile_height);
        memcpy2D(p_imag[i][1], tile_width * sizeof(double), p_imag[i][0], tile_width * sizeof(double), tile_width * sizeof(double), tile_height);
        external_pot_real[i] = _external_pot_real[i];
        external_pot_imag[i] = _external_pot_imag[i];
    }
    two_wavefunctions = true;

#ifdef HAVE_MPI
    // Halo exchange uses wave pattern to communicate
    // halo_x-wide inner rows are sent first to left and right
    // Then full length rows are exchanged to the top and bottom
    int count = inner_end_y - inner_start_y;    // The number of rows in the halo submatrix
    int block_length = halo_x;  // The number of columns in the halo submatrix
    int stride = tile_width;    // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &verticalBorder);
    MPI_Type_commit (&verticalBorder);

    count = halo_y; // The vertical halo in rows
    block_length = tile_width;  // The number of columns of the matrix
    stride = tile_width;    // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &horizontalBorder);
    MPI_Type_commit (&horizontalBorder);
#endif
}

void CPUBlock::update_potential(double *_external_pot_real, double *_external_pot_imag, int which) {
    external_pot_real[which] = _external_pot_real;
    external_pot_imag[which] = _external_pot_imag;
}

CPUBlock::~CPUBlock() {
    delete [] p_real[0][1];
    delete [] p_imag[0][1];
    delete [] p_real[1][1];
    delete [] p_imag[1][1];
    delete [] aH;
    delete [] bH;
    delete [] aV;
    delete [] bV;
    delete [] norm;
    delete [] coupling_const;
    delete [] LeeHuangYang_coupling;
}

void CPUBlock::run_kernel() {
    // Inner part
    int inner = 1, sides = 0;
    if (halo_y == 0) {
        process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y,
                     alpha_x, alpha_y, tile_width, block_width, block_height,
                     halo_x, 0, block_height, halo_y, block_height - 2 * halo_y,
                     aH[state_index], bH[state_index], aV[state_index], bV[state_index], kin_radial[state_index],
                     coupling_const[state_index], coupling_const[2], LeeHuangYang_coupling[state_index],
                     external_pot_real[state_index], external_pot_imag[state_index],
                     p_real[state_index][sense], p_imag[state_index][sense],
                     p_real[1 - state_index][sense], p_imag[1 - state_index][sense],
                     p_real[state_index][1 - sense], p_imag[state_index][1 - sense],
                     inner, sides, imag_time, coordinate_system);

    }
    else {
#ifndef HAVE_MPI
        #pragma omp parallel default(shared)
#endif
        {
#ifndef HAVE_MPI
            #pragma omp for schedule(dynamic)
#endif
            for (int block_start = block_height - 2 * halo_y;
            block_start < int(tile_height - block_height);
            block_start += block_height - 2 * halo_y) {

                process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y,
                alpha_x, alpha_y, tile_width, block_width, block_height,
                halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y,
                aH[state_index], bH[state_index], aV[state_index], bV[state_index], kin_radial[state_index],
                coupling_const[state_index], coupling_const[2], LeeHuangYang_coupling[state_index],
                external_pot_real[state_index], external_pot_imag[state_index],
                p_real[state_index][sense], p_imag[state_index][sense],
                p_real[1 - state_index][sense], p_imag[1 - state_index][sense],
                p_real[state_index][1 - sense], p_imag[state_index][1 - sense],
                inner, sides, imag_time, coordinate_system);
            }
        }
    }
    sense = 1 - sense;
}

void CPUBlock::run_kernel_on_halo() {
    int inner = 0, sides = 0;
    if (tile_height <= block_height) {
        // One full band
        inner = 1;
        sides = 1;
        process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y,
                     alpha_x, alpha_y, tile_width, block_width, block_height,
                     halo_x, 0, tile_height, 0, tile_height,
                     aH[state_index], bH[state_index], aV[state_index], bV[state_index], kin_radial[state_index],
                     coupling_const[state_index], coupling_const[2], LeeHuangYang_coupling[state_index],
                     external_pot_real[state_index], external_pot_imag[state_index],
                     p_real[state_index][sense], p_imag[state_index][sense],
                     p_real[1 - state_index][sense], p_imag[1 - state_index][sense],
                     p_real[state_index][1 - sense], p_imag[state_index][1 - sense],
                     inner, sides, imag_time, coordinate_system);
    }
    else {

        // Sides
        inner = 0;
        sides = 1;
#ifndef HAVE_MPI
        #pragma omp parallel for schedule(dynamic)
#endif
        for (int block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {
            process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y,
                         alpha_x, alpha_y, tile_width, block_width, block_height,
                         halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y,
                         aH[state_index], bH[state_index], aV[state_index], bV[state_index], kin_radial[state_index],
                         coupling_const[state_index], coupling_const[2], LeeHuangYang_coupling[state_index],
                         external_pot_real[state_index], external_pot_imag[state_index],
                         p_real[state_index][sense], p_imag[state_index][sense],
                         p_real[1 - state_index][sense], p_imag[1 - state_index][sense],
                         p_real[state_index][1 - sense], p_imag[state_index][1 - sense],
                         inner, sides, imag_time, coordinate_system);
        }
        size_t block_start;
        for (block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {}
        // First band
        inner = 1;
        sides = 1;
        process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y,
                     alpha_x, alpha_y, tile_width, block_width, block_height,
                     halo_x, 0, block_height, 0, block_height - halo_y,
                     aH[state_index], bH[state_index], aV[state_index], bV[state_index], kin_radial[state_index],
                     coupling_const[state_index], coupling_const[2], LeeHuangYang_coupling[state_index],
                     external_pot_real[state_index], external_pot_imag[state_index],
                     p_real[state_index][sense], p_imag[state_index][sense],
                     p_real[1 - state_index][sense], p_imag[1 - state_index][sense],
                     p_real[state_index][1 - sense], p_imag[state_index][1 - sense],
                     inner, sides, imag_time, coordinate_system);

        // Last band
        inner = 1;
        sides = 1;
        process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y,
                     alpha_x, alpha_y, tile_width, block_width, block_height,
                     halo_x, block_start, tile_height - block_start, halo_y, tile_height - block_start - halo_y,
                     aH[state_index], bH[state_index], aV[state_index], bV[state_index], kin_radial[state_index],
                     coupling_const[state_index], coupling_const[2], LeeHuangYang_coupling[state_index],
                     external_pot_real[state_index], external_pot_imag[state_index],
                     p_real[state_index][sense], p_imag[state_index][sense],
                     p_real[1 - state_index][sense], p_imag[1 - state_index][sense],
                     p_real[state_index][1 - sense], p_imag[state_index][1 - sense],
                     inner, sides, imag_time, coordinate_system);
    }
}

double CPUBlock::calculate_squared_norm(bool global) const {
    double norm2 = 0.;
#ifndef HAVE_MPI
    #pragma omp parallel for reduction(+:norm2) schedule(dynamic, 8)
#endif
    for(int i = inner_start_y - start_y; i < inner_end_y - start_y; i++) {
        for(int j = inner_start_x - start_x; j < inner_end_x - start_x; j++) {
            norm2 += p_real[state_index][sense][j + i * tile_width] * p_real[state_index][sense][j + i * tile_width] + p_imag[state_index][sense][j + i * tile_width] * p_imag[state_index][sense][j + i * tile_width];
        }
    }
#ifdef HAVE_MPI
    if (global) {
        int nProcs = 1;
        MPI_Comm_size(cartcomm, &nProcs);
        double *sums = new double[nProcs];
        MPI_Allgather(&norm2, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
        norm2 = 0.;
        for(int i = 0; i < nProcs; i++)
            norm2 += sums[i];
        delete [] sums;
    }
#endif
    return norm2 * delta_x * delta_y;
}

void CPUBlock::wait_for_completion() {
    if (imag_time && norm[state_index] != 0) {
        //normalization
        double tot_norm = calculate_squared_norm(true);
        double _norm = sqrt(tot_norm / norm[state_index]);

        for (size_t i = 0; i < tile_height; i++) {
            for (size_t j = 0; j < tile_width; j++) {
                p_real[state_index][sense][j + i * tile_width] /= _norm;
                p_imag[state_index][sense][j + i * tile_width] /= _norm;
            }
        }
    }
    if (two_wavefunctions) {
        if (state_index == 0) {
            sense = 1 - sense;
        }
        state_index = 1 - state_index;
    }
}

void CPUBlock::get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag, double *dest_real2, double * dest_imag2) const {
    memcpy2D(dest_real, dest_stride * sizeof(double), &(p_real[0][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
    memcpy2D(dest_imag, dest_stride * sizeof(double), &(p_imag[0][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
    if (dest_real2 != 0) {
        memcpy2D(dest_real2, dest_stride * sizeof(double), &(p_real[1][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
        memcpy2D(dest_imag2, dest_stride * sizeof(double), &(p_imag[1][sense][y * tile_width + x]), tile_width * sizeof(double), width * sizeof(double), height);
    }
}

void CPUBlock::rabi_coupling(double var, double delta_t) {
    double norm_omega = sqrt(coupling_const[3] * coupling_const[3] + coupling_const[4] * coupling_const[4]);
    double cc, cs_r, cs_i;
    if(imag_time) {
        cc = cosh(- delta_t * var * norm_omega);
        if (norm_omega == 0) {
            cs_r = 0;
            cs_i = 0;
        }
        else {
            cs_r = coupling_const[3] / norm_omega * sinh(- delta_t * var * norm_omega);
            cs_i = coupling_const[4] / norm_omega * sinh(- delta_t * var * norm_omega);
        }
        rabi_coupling_imaginary(tile_width, tile_width, tile_height, cc, cs_r, cs_i, p_real[0][sense], p_imag[0][sense], p_real[1][sense], p_imag[1][sense]);
    }
    else {
        cc = cos(- delta_t * var * norm_omega);
        if (norm_omega == 0) {
            cs_r = 0;
            cs_i = 0;
        }
        else {
            cs_r = coupling_const[3] / norm_omega * sin(- delta_t * var * norm_omega);
            cs_i = coupling_const[4] / norm_omega * sin(- delta_t * var * norm_omega);
        }
        rabi_coupling_real(tile_width, tile_width, tile_height, cc, cs_r, cs_i, p_real[0][sense], p_imag[0][sense], p_real[1][sense], p_imag[1][sense]);
    }
}

void CPUBlock::normalization() {
    if(imag_time && (coupling_const[3] != 0 || coupling_const[4] != 0)) {
        //normalization
        int nProcs = 1;
#ifdef HAVE_MPI
        MPI_Comm_size(cartcomm, &nProcs);
#endif

        double sum_a = 0., sum_b = 0., *sums, *sums_a, *sums_b;
        sums = new double[nProcs];
        sums_a = new double[nProcs];
        sums_b = new double[nProcs];
        for(int i = inner_start_y - start_y; i < inner_end_y - start_y; i++) {
            for(int j = inner_start_x - start_x; j < inner_end_x - start_x; j++) {
                sum_a += p_real[0][sense][j + i * tile_width] * p_real[0][sense][j + i * tile_width] + p_imag[0][sense][j + i * tile_width] * p_imag[0][sense][j + i * tile_width];
            }
        }
        if(p_real[1] != NULL) {
            for(int i = inner_start_y - start_y; i < inner_end_y - start_y; i++) {
                for(int j = inner_start_x - start_x; j < inner_end_x - start_x; j++) {
                    sum_b += p_real[1][sense][j + i * tile_width] * p_real[1][sense][j + i * tile_width] + p_imag[1][sense][j + i * tile_width] * p_imag[1][sense][j + i * tile_width];
                }
            }
        }
#ifdef HAVE_MPI

        MPI_Allgather(&sum_a, 1, MPI_DOUBLE, sums_a, 1, MPI_DOUBLE, cartcomm);
        MPI_Allgather(&sum_b, 1, MPI_DOUBLE, sums_b, 1, MPI_DOUBLE, cartcomm);
#else
        sums_a[0] = sum_a;
        sums_b[0] = sum_b;
#endif
        double tot_sum_a = 0., tot_sum_b = 0.;
        for(int i = 0; i < nProcs; i++) {
            tot_sum_a += sums_a[i];
            tot_sum_b += sums_b[i];
        }
        double _norm = sqrt((tot_sum_a + tot_sum_b) * delta_x * delta_y / tot_norm);

        for(size_t i = 0; i < tile_height; i++) {
            for(size_t j = 0; j < tile_width; j++) {
                p_real[0][sense][j + i * tile_width] /= _norm;
                p_imag[0][sense][j + i * tile_width] /= _norm;
            }
        }
        norm[0] = tot_sum_a / (tot_sum_a + tot_sum_b) * tot_norm;
        if(p_real[1] != NULL) {
            for(size_t i = 0; i < tile_height; i++) {
                for(size_t j = 0; j < tile_width; j++) {
                    p_real[1][sense][j + i * tile_width] /= _norm;
                    p_imag[1][sense][j + i * tile_width] /= _norm;
                }
            }
            norm[1] = tot_sum_b / (tot_sum_a + tot_sum_b) * tot_norm;
        }
        delete[] sums;
    }
}

void CPUBlock::cpy_first_positive_to_first_negative() {
    if (imag_time && coordinate_system == "cylindrical") {
        // performs the copy only for the tiles containing the origin of the radial coordinate
        if (start_x <= 0) {
            int stride = end_x - start_x;
            double sign;
            sign = (angular_momentum[0] % 2 == 0 ? 1 : -1);
            for (int j = start_y, idx = 1, peer = 0; j < end_y; j += 1, idx += stride, peer += stride) {
                p_real[0][sense][peer] = sign * p_real[0][sense][idx];
                p_imag[0][sense][peer] = sign * p_imag[0][sense][idx];
            }
            if (two_wavefunctions) {
                sign = (angular_momentum[1] % 2 == 0 ? 1 : -1);
                for (int j = start_y, idx = 1, peer = 0; j < end_y; j += 1, idx += stride, peer += stride) {
                    p_real[1][sense][peer] = sign * p_real[1][sense][idx];
                    p_imag[1][sense][peer] = sign * p_imag[1][sense][idx];
                }
            }
        }
    }
}

void CPUBlock::start_halo_exchange() {
    // Halo exchange: LEFT/RIGHT
#ifdef HAVE_MPI
    int offset = (inner_start_y - start_y) * tile_width;
    MPI_Irecv(p_real[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 1, cartcomm, req);
    MPI_Irecv(p_imag[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 2, cartcomm, req + 1);
    offset = (inner_start_y - start_y) * tile_width + inner_end_x - start_x;
    MPI_Irecv(p_real[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 3, cartcomm, req + 2);
    MPI_Irecv(p_imag[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 4, cartcomm, req + 3);

    offset = (inner_start_y - start_y) * tile_width + inner_end_x - halo_x - start_x;
    MPI_Isend(p_real[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 1, cartcomm, req + 4);
    MPI_Isend(p_imag[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 2, cartcomm, req + 5);
    offset = (inner_start_y - start_y) * tile_width + halo_x;
    MPI_Isend(p_real[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 3, cartcomm, req + 6);
    MPI_Isend(p_imag[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 4, cartcomm, req + 7);
#else
    if(periods[1] != 0) {
        int offset = (inner_start_y - start_y) * tile_width;
        memcpy2D(&(p_real[state_index][1 - sense][offset]), tile_width * sizeof(double), &(p_real[state_index][1 - sense][offset + tile_width - 2 * halo_x]), tile_width * sizeof(double), halo_x * sizeof(double), tile_height - 2 * halo_y);
        memcpy2D(&(p_imag[state_index][1 - sense][offset]), tile_width * sizeof(double), &(p_imag[state_index][1 - sense][offset + tile_width - 2 * halo_x]), tile_width * sizeof(double), halo_x * sizeof(double), tile_height - 2 * halo_y);
        memcpy2D(&(p_real[state_index][1 - sense][offset + tile_width - halo_x]), tile_width * sizeof(double), &(p_real[state_index][1 - sense][offset + halo_x]), tile_width * sizeof(double), halo_x * sizeof(double), tile_height - 2 * halo_y);
        memcpy2D(&(p_imag[state_index][1 - sense][offset + tile_width - halo_x]), tile_width * sizeof(double), &(p_imag[state_index][1 - sense][offset + halo_x]), tile_width * sizeof(double), halo_x * sizeof(double), tile_height - 2 * halo_y);
    }
#endif
}

void CPUBlock::finish_halo_exchange() {
#ifdef HAVE_MPI
    MPI_Waitall(8, req, statuses);

    // Halo exchange: UP/DOWN
    int offset = 0;
    MPI_Irecv(p_real[state_index][sense] + offset, 1, horizontalBorder, neighbors[UP], 1, cartcomm, req);
    MPI_Irecv(p_imag[state_index][sense] + offset, 1, horizontalBorder, neighbors[UP], 2, cartcomm, req + 1);
    offset = (inner_end_y - start_y) * tile_width;
    MPI_Irecv(p_real[state_index][sense] + offset, 1, horizontalBorder, neighbors[DOWN], 3, cartcomm, req + 2);
    MPI_Irecv(p_imag[state_index][sense] + offset, 1, horizontalBorder, neighbors[DOWN], 4, cartcomm, req + 3);

    offset = (inner_end_y - halo_y - start_y) * tile_width;
    MPI_Isend(p_real[state_index][sense] + offset, 1, horizontalBorder, neighbors[DOWN], 1, cartcomm, req + 4);
    MPI_Isend(p_imag[state_index][sense] + offset, 1, horizontalBorder, neighbors[DOWN], 2, cartcomm, req + 5);
    offset = halo_y * tile_width;
    MPI_Isend(p_real[state_index][sense] + offset, 1, horizontalBorder, neighbors[UP], 3, cartcomm, req + 6);
    MPI_Isend(p_imag[state_index][sense] + offset, 1, horizontalBorder, neighbors[UP], 4, cartcomm, req + 7);

    MPI_Waitall(8, req, statuses);
#else
    if(periods[0] != 0) {
        int offset = (inner_end_y - start_y) * tile_width;
        memcpy2D(&(p_real[state_index][sense][0]), tile_width * sizeof(double), &(p_real[state_index][sense][offset - halo_y * tile_width]), tile_width * sizeof(double), tile_width * sizeof(double), halo_y);
        memcpy2D(&(p_imag[state_index][sense][0]), tile_width * sizeof(double), &(p_imag[state_index][sense][offset - halo_y * tile_width]), tile_width * sizeof(double), tile_width * sizeof(double), halo_y);
        memcpy2D(&(p_real[state_index][sense][offset]), tile_width * sizeof(double), &(p_real[state_index][sense][halo_y * tile_width]), tile_width * sizeof(double), tile_width * sizeof(double), halo_y);
        memcpy2D(&(p_imag[state_index][sense][offset]), tile_width * sizeof(double), &(p_imag[state_index][sense][halo_y * tile_width]), tile_width * sizeof(double), tile_width * sizeof(double), halo_y);
    }
#endif
}
