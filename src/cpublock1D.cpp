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
#include "common.h"
#include "kernel1D.h"
#include <iostream> //

// Helpers




void block_kernel_horizontal(size_t start_offset, size_t stride, size_t width, double a, double b, double * p_real, double * p_imag) {

        for (size_t idx =  (start_offset ) % 2, peer = idx + 1; idx <  width - 1; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real - b * p_imag[peer];
            p_imag[idx] = a * tmp_imag + b * p_real[peer];
            p_real[peer] = a * p_real[peer] - b * tmp_imag;
            p_imag[peer] = a * p_imag[peer] + b * tmp_real;
        }

}

void block_kernel_horizontal_imaginary(size_t start_offset, size_t stride, size_t width,  double a, double b, double * p_real, double * p_imag) {

        for (size_t idx =  (start_offset ) % 2, peer = idx + 1; idx <  width - 1; idx += 2, peer += 2) {
            double tmp_real = p_real[idx];
            double tmp_imag = p_imag[idx];
            p_real[idx] = a * tmp_real + b * p_real[peer];
            p_imag[idx] = a * tmp_imag + b * p_imag[peer];
            p_real[peer] = a * p_real[peer] + b * tmp_real;
            p_imag[peer] = a * p_imag[peer] + b * tmp_imag;
        }

}

//double time potential
void block_kernel_potential(bool two_wavefunctions, size_t stride, size_t width,  double a, double b, double coupling_a, double coupling_b, size_t tile_width,
                            const double *external_pot_real, const double *external_pot_imag, const double *pb_real, const double *pb_imag, double * p_real, double * p_imag) {
    if(two_wavefunctions) {
            for (size_t idx = 0, idx_pot = 0; idx < width; ++idx, ++idx_pot) {
                double norm_2 = p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx];
                double norm_2b = pb_real[idx_pot] * pb_real[idx_pot] + pb_imag[idx_pot] * pb_imag[idx_pot];
                double c_cos = cos(coupling_a * norm_2 + coupling_b * norm_2b);
                double c_sin = sin(coupling_a * norm_2 + coupling_b * norm_2b);
                double tmp = p_real[idx];
                p_real[idx] = external_pot_real[idx_pot] * tmp - external_pot_imag[idx_pot] * p_imag[idx];
                p_imag[idx] = external_pot_real[idx_pot] * p_imag[idx] + external_pot_imag[idx_pot] * tmp;

                tmp = p_real[idx];
                p_real[idx] = c_cos * tmp + c_sin * p_imag[idx];
                p_imag[idx] = c_cos * p_imag[idx] - c_sin * tmp;
            }
    }
    else {
            for (size_t idx = 0, idx_pot = 0; idx <  width; ++idx, ++idx_pot) {
                double norm_2 = p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx];
                double c_cos = cos(coupling_a * norm_2);
                double c_sin = sin(coupling_a * norm_2);
                double tmp = p_real[idx];
                p_real[idx] = external_pot_real[idx_pot] * tmp - external_pot_imag[idx_pot] * p_imag[idx];
                p_imag[idx] = external_pot_real[idx_pot] * p_imag[idx] + external_pot_imag[idx_pot] * tmp;

                tmp = p_real[idx];
                p_real[idx] = c_cos * tmp + c_sin * p_imag[idx];
                p_imag[idx] = c_cos * p_imag[idx] - c_sin * tmp;
            }
    }
}

//double time potential
void block_kernel_potential_imaginary(bool two_wavefunctions, size_t stride, size_t width,  double a, double b, double coupling_a, double coupling_b, size_t tile_width,
                                      const double *external_pot_real, const double *external_pot_imag, const double *pb_real, const double *pb_imag, double * p_real, double * p_imag) {
    if(two_wavefunctions) {
            for (size_t idx = 0, idx_pot = 0; idx < width; ++idx, ++idx_pot) {
                double tmp = exp(-1. * (coupling_a * (p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx]) + coupling_b * (pb_real[idx_pot] * pb_real[idx_pot] + pb_imag[idx_pot] * pb_imag[idx_pot])));
                p_real[idx] = tmp * external_pot_real[idx_pot] * p_real[idx];
                p_imag[idx] = tmp * external_pot_real[idx_pot] * p_imag[idx];
            }
    }
    else {
            for (size_t idx = 0, idx_pot = 0; idx <  width; ++idx, ++idx_pot) {
                double tmp = exp(-1. * coupling_a * (p_real[idx] * p_real[idx] + p_imag[idx] * p_imag[idx]));
                p_real[idx] = tmp * external_pot_real[idx_pot] * p_real[idx];
                p_imag[idx] = tmp * external_pot_real[idx_pot] * p_imag[idx];
            }
    }
}

//rotation


void rabi_coupling_real(size_t stride, size_t width, double cc, double cs_r, double cs_i, double *p_real, double *p_imag, double *pb_real, double *pb_imag) {
    double real, imag;
        for(size_t j = 0, idx = 0; j < width; j++, idx++) {
            real = p_real[idx];
            imag = p_imag[idx];
            p_real[idx] = cc * real - cs_i * pb_real[idx] - cs_r * pb_imag[idx];
            p_imag[idx] = cc * imag + cs_r * pb_real[idx] - cs_i * pb_imag[idx];
            pb_real[idx] = cc * pb_real[idx] + cs_i * real - cs_r * imag;
            pb_imag[idx] = cc * pb_imag[idx] + cs_r * real + cs_i * imag;
        }
}

void rabi_coupling_imaginary(size_t stride, size_t width, double cc, double cs_r, double cs_i, double *p_real, double *p_imag, double *pb_real, double *pb_imag) {
    double real, imag;
        for(size_t j = 0, idx =0; j < width; j++, idx++) {
            real = p_real[idx];
            imag = p_imag[idx];
            p_real[idx] = cc * real + cs_r * pb_real[idx] - cs_i * pb_imag[idx];
            p_imag[idx] = cc * imag + cs_i * pb_real[idx] + cs_r * pb_imag[idx];
            pb_real[idx] = cc * pb_real[idx] + cs_r * real + cs_i * imag;
            pb_imag[idx] = cc * pb_imag[idx] - cs_i * real + cs_r * imag;
        }
}

void full_step(bool two_wavefunctions, size_t stride, size_t width, int offset_x, double a, double b, double coupling_a, double coupling_b,
               size_t tile_width, const double *external_pot_real, const double *external_pot_imag, const double *pb_real, const double *pb_imag, double * real, double * imag) {

    block_kernel_horizontal(0u, stride, width, a, b, real, imag);
	
    block_kernel_horizontal(1u, stride, width, a, b, real, imag);
	
    block_kernel_potential (two_wavefunctions, stride, width,  a, b, coupling_a, coupling_b, tile_width, external_pot_real, external_pot_imag, pb_real, pb_imag, real, imag);

    block_kernel_horizontal(1u, stride, width, a, b, real, imag);
	
    block_kernel_horizontal(0u, stride, width, a, b, real, imag);

}

void full_step_imaginary(bool two_wavefunctions, size_t stride, size_t width, int offset_x, double a, double b, double coupling_a, double coupling_b,
                         size_t tile_width, const double *external_pot_real, const double *external_pot_imag, const double *pb_real, const double *pb_imag, double * real, double * imag) {

    block_kernel_horizontal_imaginary(0u, stride, width, a, b, real, imag);
    block_kernel_horizontal_imaginary(1u, stride, width, a, b, real, imag);
    block_kernel_potential_imaginary (two_wavefunctions, stride, width,  a, b, coupling_a, coupling_b, tile_width, external_pot_real, external_pot_imag, pb_real, pb_imag, real, imag);

    block_kernel_horizontal_imaginary(1u, stride, width,  a, b, real, imag);
    block_kernel_horizontal_imaginary(0u, stride, width,  a, b, real, imag);
}

void process_sides(bool two_wavefunctions, int offset_tile_x, size_t tile_width, size_t block_width, size_t halo_x, 
                   double a, double b, double coupling_a, double coupling_b, const double *external_pot_real, const double *external_pot_imag,
                   const double * p_real, const double * p_imag, const double * pb_real, const double * pb_imag,
                   double * next_real, double * next_imag, double * block_real, double * block_imag, bool imag_time) {

    // First block [0..block_width - halo_x]
    memcpy2D(block_real, block_width * sizeof(double), &p_real[0], tile_width * sizeof(double), block_width * sizeof(double),1 );
    memcpy2D(block_imag, block_width * sizeof(double), &p_imag[0], tile_width * sizeof(double), block_width * sizeof(double),1 );
    if(imag_time)
        full_step_imaginary(two_wavefunctions, block_width, block_width,  offset_tile_x,  a, b, coupling_a, coupling_b, tile_width,
                            &external_pot_real[0], &external_pot_imag[0], &pb_real[0], &pb_imag[0], block_real, block_imag);
    else
        full_step(two_wavefunctions, block_width, block_width, offset_tile_x, a, b, coupling_a, coupling_b, tile_width,
                  &external_pot_real[0], &external_pot_imag[0], &pb_real[0], &pb_imag[0], block_real, block_imag);
    memcpy2D(&next_real[0], tile_width * sizeof(double), &block_real[0], block_width * sizeof(double), (block_width - halo_x) * sizeof(double),1 );
    memcpy2D(&next_imag[0], tile_width * sizeof(double), &block_imag[0], block_width * sizeof(double), (block_width - halo_x) * sizeof(double),1 );

    size_t block_start = ((tile_width - block_width) / (block_width - 2 * halo_x) + 1) * (block_width - 2 * halo_x);
    // Last block
    memcpy2D(block_real, block_width * sizeof(double), &p_real[ block_start], tile_width * sizeof(double), (tile_width - block_start) * sizeof(double),1 );
    memcpy2D(block_imag, block_width * sizeof(double), &p_imag[ block_start], tile_width * sizeof(double), (tile_width - block_start) * sizeof(double),1 );
    if(imag_time)
        full_step_imaginary(two_wavefunctions, block_width, tile_width - block_start,  offset_tile_x + block_start, a, b, coupling_a, coupling_b, tile_width,
                            &external_pot_real[ block_start], &external_pot_imag[ block_start], &pb_real[ block_start], &pb_imag[ block_start], block_real, block_imag);
    else
        full_step(two_wavefunctions, block_width, tile_width - block_start, offset_tile_x + block_start, a, b, coupling_a, coupling_b, tile_width,
                  &external_pot_real[ block_start], &external_pot_imag[ block_start], &pb_real[ block_start], &pb_imag[ block_start], block_real, block_imag);
    memcpy2D(&next_real[ block_start + halo_x], tile_width * sizeof(double), &block_real[ halo_x], block_width * sizeof(double), (tile_width - block_start - halo_x) * sizeof(double),1 );
    memcpy2D(&next_imag[ block_start + halo_x], tile_width * sizeof(double), &block_imag[ halo_x], block_width * sizeof(double), (tile_width - block_start - halo_x) * sizeof(double),1 );
}

void process_band(bool two_wavefunctions, int offset_tile_x, size_t tile_width, size_t block_width, size_t halo_x, 
                  double a, double b, double coupling_a, double coupling_b, const double *external_pot_real, const double *external_pot_imag, const double * p_real, const double * p_imag,
                  const double * pb_real, const double * pb_imag, double * next_real, double * next_imag, int inner, int sides, bool imag_time) {
    double *block_real = new double[ block_width];
    double *block_imag = new double[ block_width];

    if (tile_width <= block_width) {
        if (sides) {
            // One full block
            memcpy2D(block_real, block_width * sizeof(double), &p_real[0], tile_width * sizeof(double), tile_width * sizeof(double),1 );
			
			memcpy2D(block_imag, block_width * sizeof(double), &p_imag[0], tile_width * sizeof(double), tile_width * sizeof(double),1 );
			
			if(imag_time)
                full_step_imaginary(two_wavefunctions, block_width, tile_width, offset_tile_x, a, b, coupling_a, coupling_b, tile_width,
                                    &external_pot_real[0], &external_pot_imag[0], &pb_real[0], &pb_imag[0], block_real, block_imag);
            else
				
                full_step(two_wavefunctions, block_width, tile_width, offset_tile_x,  a, b, coupling_a, coupling_b, tile_width,
                          &external_pot_real[0], &external_pot_imag[0], &pb_real[0], &pb_imag[0], block_real, block_imag);
			
			
		    memcpy2D(&next_real[0], tile_width * sizeof(double), &block_real[0], block_width * sizeof(double), tile_width * sizeof(double),1 );
			
			memcpy2D(&next_imag[0], tile_width * sizeof(double), &block_imag[0], block_width * sizeof(double), tile_width * sizeof(double),1 );
        }
    }
    else {
        if (sides) {
            process_sides(two_wavefunctions, offset_tile_x, tile_width, block_width, halo_x, a, b, coupling_a, coupling_b, external_pot_real, external_pot_imag, p_real, p_imag, pb_real, pb_imag, next_real, next_imag, block_real, block_imag, imag_time);
			
		}
        if (inner) {
            for (size_t block_start = block_width - 2 * halo_x; block_start < tile_width - block_width; block_start += block_width - 2 * halo_x) {
                memcpy2D(block_real, block_width * sizeof(double), &p_real[ block_start], tile_width * sizeof(double), block_width * sizeof(double),1 );
                memcpy2D(block_imag, block_width * sizeof(double), &p_imag[ block_start], tile_width * sizeof(double), block_width * sizeof(double),1 );
                if(imag_time)
                    full_step_imaginary(two_wavefunctions, block_width, block_width,  offset_tile_x + block_start,  a, b, coupling_a, coupling_b, tile_width,
                                        &external_pot_real[ block_start], &external_pot_imag[ block_start], &pb_real[ block_start], &pb_imag[ block_start], block_real, block_imag);
                else
                    full_step(two_wavefunctions, block_width, block_width,  offset_tile_x + block_start, a, b, coupling_a, coupling_b, tile_width,
                              &external_pot_real[ block_start], &external_pot_imag[ block_start], &pb_real[ block_start], &pb_imag[ block_start], block_real, block_imag);
				
				memcpy2D(&next_real[ block_start + halo_x], tile_width * sizeof(double), &block_real[ halo_x], block_width * sizeof(double), (block_width - 2 * halo_x) * sizeof(double),1 );
                memcpy2D(&next_imag[ block_start + halo_x], tile_width * sizeof(double), &block_imag[ halo_x], block_width * sizeof(double), (block_width - 2 * halo_x) * sizeof(double),1 );
            }
        }
    }


    delete[] block_real;
    delete[] block_imag;
}

// Class methods
CPUBlock1D::CPUBlock1D(Lattice1D *grid, State1D *state, Hamiltonian1D *hamiltonian,
                   double *_external_pot_real, double *_external_pot_imag,
                   double _a, double _b, double delta_t,
                   double _norm, bool _imag_time):
    sense(0),
    state_index(0),
    imag_time(_imag_time) {
    delta_x = grid->delta_x;
    halo_x = grid->halo_x;
    periods = grid->periods;
    rot_coord_x = hamiltonian->rot_coord_x;
    a = new double [1];
    b = new double [1];
    coupling_const = new double [3];
    norm = new double [1];
    a[0] = _a;
    b[0] = _b;
    coupling_const[0] = hamiltonian->coupling_a * delta_t;
    coupling_const[1] = 0.;
    coupling_const[2] = 0.;
    norm[0] = _norm;
    tot_norm = norm[0];
#ifdef HAVE_MPI
    cartcomm = grid->cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
#endif

    start_x = grid->start_x;
    end_x = grid->end_x;
    inner_start_x = grid->inner_start_x;
    inner_end_x = grid->inner_end_x;
    tile_width = end_x - start_x;

    p_real[0][0] = state->p_real;
    p_imag[0][0] = state->p_imag;
    p_real[0][1] = new double[tile_width];
    p_imag[0][1] = new double[tile_width];
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
   // int count = inner_end_y - inner_start_y;  // The number of rows in the halo submatrix
    int block_length = halo_x;  // The number of columns in the halo submatrix
    int stride = tile_width;  // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &verticalBorder);
    MPI_Type_commit (&verticalBorder);

    //count = halo_y; // The vertical halo in rows
    block_length = tile_width;  // The number of columns of the matrix
    stride = tile_width;  // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &horizontalBorder);
    MPI_Type_commit (&horizontalBorder);
#endif
}

CPUBlock1D::CPUBlock1D(Lattice1D *grid, State1D *state1, State1D *state2,
                   Hamiltonian2Component1D *hamiltonian,
                   double **_external_pot_real, double **_external_pot_imag,
                   double *_a, double *_b, double delta_t,
                   double *_norm, bool _imag_time):
    sense(0),
    state_index(0),
    imag_time(_imag_time) {
    delta_x = grid->delta_x;
    halo_x = grid->halo_x;
    rot_coord_x = hamiltonian->rot_coord_x;
    a = _a;                                       //coupling 3, 4???
    b = _b;
    norm = _norm;
    tot_norm = norm[0] + norm[1];
    coupling_const = new double[5];
    coupling_const[0] = delta_t * hamiltonian->coupling_a;
    coupling_const[1] = delta_t * hamiltonian->coupling_b;
    coupling_const[2] = delta_t * hamiltonian->coupling_ab;
    coupling_const[3] = 0.5 * hamiltonian->omega_r;
    coupling_const[4] = 0.5 * hamiltonian->omega_i;
    periods = grid->periods;

#ifdef HAVE_MPI
    cartcomm = grid->cartcomm;
    MPI_Cart_shift(cartcomm, 0, 1, &neighbors[UP], &neighbors[DOWN]);
    MPI_Cart_shift(cartcomm, 1, 1, &neighbors[LEFT], &neighbors[RIGHT]);
#endif

    start_x = grid->start_x;
    end_x = grid->end_x;
    inner_start_x = grid->inner_start_x;
    inner_end_x = grid->inner_end_x;
    tile_width = end_x - start_x;
    p_real[0][0] = state1->p_real;
    p_imag[0][0] = state1->p_imag;
    p_real[1][0] = state2->p_real;
    p_imag[1][0] = state2->p_imag;

    for(int i = 0; i < 2; i++) {
        p_real[i][1] = new double[tile_width];
        p_imag[i][1] = new double[tile_width];
        memcpy2D(p_real[i][1], tile_width * sizeof(double), p_real[i][0], tile_width * sizeof(double), tile_width * sizeof(double),1);
        memcpy2D(p_imag[i][1], tile_width * sizeof(double), p_imag[i][0], tile_width * sizeof(double), tile_width * sizeof(double),1);
        external_pot_real[i] = _external_pot_real[i];
        external_pot_imag[i] = _external_pot_imag[i];
    }
    two_wavefunctions = true;

#ifdef HAVE_MPI
    // Halo exchange uses wave pattern to communicate
    // halo_x-wide inner rows are sent first to left and right
    // Then full length rows are exchanged to the top and bottom
    //int count = inner_end_y - inner_start_y;    // The number of rows in the halo submatrix
    int block_length = halo_x;  // The number of columns in the halo submatrix
    int stride = tile_width;    // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &verticalBorder);
    MPI_Type_commit (&verticalBorder);

    //count = halo_y; // The vertical halo in rows
    block_length = tile_width;  // The number of columns of the matrix
    stride = tile_width;    // The combined width of the matrix with the halo
    MPI_Type_vector (count, block_length, stride, MPI_DOUBLE, &horizontalBorder);
    MPI_Type_commit (&horizontalBorder);
#endif
}

void CPUBlock1D::update_potential(double *_external_pot_real, double *_external_pot_imag) {
    external_pot_real[0] = _external_pot_real;
    external_pot_imag[0] = _external_pot_imag;
}

CPUBlock1D::~CPUBlock1D() {
    delete[] p_real[0][1];
    delete[] p_imag[0][1];
    delete[] p_real[1][1];
    delete[] p_imag[1][1];
    if (!two_wavefunctions) {
        delete [] a;
        delete [] b;
        delete [] norm;
    }
    delete [] coupling_const;
}

void CPUBlock1D::run_kernel() {
    // Inner part
    int inner = 1, sides = 0;
#ifndef HAVE_MPI
    #pragma omp parallel default(shared)
#endif
    {
#ifndef HAVE_MPI
        #pragma omp for
#endif
            process_band(two_wavefunctions, start_x - rot_coord_x, tile_width, block_width, halo_x, a[state_index], b[state_index],
                         coupling_const[state_index], coupling_const[2], external_pot_real[state_index], external_pot_imag[state_index], p_real[state_index][sense], p_imag[state_index][sense], p_real[1 - state_index][sense], p_imag[1 - state_index][sense], p_real[state_index][1 - sense], p_imag[state_index][1 - sense], inner, sides, imag_time);
    }

    sense = 1 - sense;
}

void CPUBlock1D::run_kernel_on_halo() {
    int inner = 0, sides = 0;
        // One full band
        inner = 1;
        sides = 1;
        process_band(two_wavefunctions, start_x - rot_coord_x, tile_width, block_width, halo_x, a[state_index], b[state_index],
                     coupling_const[state_index], coupling_const[2], external_pot_real[state_index], external_pot_imag[state_index], p_real[state_index][sense], p_imag[state_index][sense],
                     p_real[1 - state_index][sense], p_imag[1 - state_index][sense], p_real[state_index][1 - sense], p_imag[state_index][1 - sense], inner, sides, imag_time);

	/*
    else {

        // Sides
        inner = 0;
        sides = 1;
#ifndef HAVE_MPI
        #pragma omp parallel for
#endif
        for (int block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {
            process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y, alpha_x, alpha_y, tile_width, block_width, block_height, halo_x, block_start, block_height, halo_y, block_height - 2 * halo_y, a[state_index], b[state_index],
                         coupling_const[state_index], coupling_const[2], external_pot_real[state_index], external_pot_imag[state_index], p_real[state_index][sense], p_imag[state_index][sense],
                         p_real[1 - state_index][sense], p_imag[1 - state_index][sense], p_real[state_index][1 - sense], p_imag[state_index][1 - sense], inner, sides, imag_time);
        }
        size_t block_start;
        for (block_start = block_height - 2 * halo_y; block_start < tile_height - block_height; block_start += block_height - 2 * halo_y) {}
        // First band
        inner = 1;
        sides = 1;
        process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y, alpha_x, alpha_y, tile_width, block_width, block_height, halo_x, 0, block_height, 0, block_height - halo_y, a[state_index], b[state_index],
                     coupling_const[state_index], coupling_const[2], external_pot_real[state_index], external_pot_imag[state_index], p_real[state_index][sense], p_imag[state_index][sense],
                     p_real[1 - state_index][sense], p_imag[1 - state_index][sense], p_real[state_index][1 - sense], p_imag[state_index][1 - sense], inner, sides, imag_time);

        // Last band
        inner = 1;
        sides = 1;
        process_band(two_wavefunctions, start_x - rot_coord_x, start_y - rot_coord_y, alpha_x, alpha_y, tile_width, block_width, block_height, halo_x, block_start, tile_height - block_start, halo_y, tile_height - block_start - halo_y, a[state_index], b[state_index],
                     coupling_const[state_index], coupling_const[2], external_pot_real[state_index], external_pot_imag[state_index], p_real[state_index][sense], p_imag[state_index][sense],
                     p_real[1 - state_index][sense], p_imag[1 - state_index][sense], p_real[state_index][1 - sense], p_imag[state_index][1 - sense], inner, sides, imag_time);
    }
	*/
}

double CPUBlock1D::calculate_squared_norm(bool global) const {
    double norm2 = 0.;
#ifndef HAVE_MPI
    #pragma omp parallel for reduction(+:norm2)
#endif
        for(int j = inner_start_x - start_x; j < inner_end_x - start_x; j++) {
            norm2 += p_real[state_index][sense][j] * p_real[state_index][sense][j] + p_imag[state_index][sense][j] * p_imag[state_index][sense][j];
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
    return norm2 * delta_x;
}

void CPUBlock1D::wait_for_completion() {
    if (imag_time && norm[state_index] != 0) {
        //normalization
        double tot_norm = calculate_squared_norm(true);
        double _norm = sqrt(tot_norm / norm[state_index]);

            for (size_t j = 0; j < tile_width; j++) {
                p_real[state_index][sense][j] /= _norm;
                p_imag[state_index][sense][j] /= _norm;
            }
    }
    if (two_wavefunctions) {
        if (state_index == 0) {
            sense = 1 - sense;
        }
        state_index = 1 - state_index;
    }
}

void CPUBlock1D::get_sample(size_t dest_stride, size_t x, size_t width, double * dest_real, double * dest_imag, double *dest_real2, double * dest_imag2) const {
    memcpy2D(dest_real, dest_stride * sizeof(double), &(p_real[0][sense][ x]), tile_width * sizeof(double), width * sizeof(double),1);
    memcpy2D(dest_imag, dest_stride * sizeof(double), &(p_imag[0][sense][ x]), tile_width * sizeof(double), width * sizeof(double),1);
    if (dest_real2 != 0) {
        memcpy2D(dest_real2, dest_stride * sizeof(double), &(p_real[1][sense][ x]), tile_width * sizeof(double), width * sizeof(double),1 );
        memcpy2D(dest_imag2, dest_stride * sizeof(double), &(p_imag[1][sense][ x]), tile_width * sizeof(double), width * sizeof(double),1 );
    }
}

void CPUBlock1D::rabi_coupling(double var, double delta_t) {
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
        rabi_coupling_imaginary(tile_width, tile_width,  cc, cs_r, cs_i, p_real[0][sense], p_imag[0][sense], p_real[1][sense], p_imag[1][sense]);
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
        rabi_coupling_real(tile_width, tile_width,  cc, cs_r, cs_i, p_real[0][sense], p_imag[0][sense], p_real[1][sense], p_imag[1][sense]);
    }
}

void CPUBlock1D::normalization() {
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
            for(int j = inner_start_x - start_x; j < inner_end_x - start_x; j++) {
                sum_a += p_real[0][sense][j] * p_real[0][sense][j ] + p_imag[0][sense][j ] * p_imag[0][sense][j ];
            }

        if(p_real[1] != NULL) {

                for(int j = inner_start_x - start_x; j < inner_end_x - start_x; j++) {
                    sum_b += p_real[1][sense][j ] * p_real[1][sense][j ] + p_imag[1][sense][j ] * p_imag[1][sense][j ];
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
        double _norm = sqrt((tot_sum_a + tot_sum_b) * delta_x  / tot_norm);

            for(size_t j = 0; j < tile_width; j++) {
                p_real[0][sense][j ] /= _norm;
                p_imag[0][sense][j ] /= _norm;
            }

        norm[0] = tot_sum_a / (tot_sum_a + tot_sum_b) * tot_norm;
        if(p_real[1] != NULL) {
                for(size_t j = 0; j < tile_width; j++) {
                    p_real[1][sense][j ] /= _norm;
                    p_imag[1][sense][j ] /= _norm;
                }
            norm[1] = tot_sum_b / (tot_sum_a + tot_sum_b) * tot_norm;
        }
        delete[] sums;
    }
}

void CPUBlock1D::start_halo_exchange() {
    // Halo exchange: LEFT/RIGHT
#ifdef HAVE_MPI
    //int offset = (inner_start_y - start_y) * tile_width;
    MPI_Irecv(p_real[state_index][1 - sense] /*+ offset*/, 1, verticalBorder, neighbors[LEFT], 1, cartcomm, req);
    MPI_Irecv(p_imag[state_index][1 - sense] /*+ offset*/, 1, verticalBorder, neighbors[LEFT], 2, cartcomm, req + 1);
    int offset = /*(inner_start_y - start_y) * tile_width +*/ inner_end_x - start_x;
    MPI_Irecv(p_real[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 3, cartcomm, req + 2);
    MPI_Irecv(p_imag[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 4, cartcomm, req + 3);

    offset = /*(inner_start_y - start_y) * tile_width +*/ inner_end_x - halo_x - start_x;
    MPI_Isend(p_real[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 1, cartcomm, req + 4);
    MPI_Isend(p_imag[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[RIGHT], 2, cartcomm, req + 5);
    offset = /*(inner_start_y - start_y) * tile_width +*/ halo_x;
    MPI_Isend(p_real[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 3, cartcomm, req + 6);
    MPI_Isend(p_imag[state_index][1 - sense] + offset, 1, verticalBorder, neighbors[LEFT], 4, cartcomm, req + 7);
#else

	if (periods[0] != 0) {
        memcpy2D(&(p_real[state_index][1 - sense][0]), tile_width * sizeof(double), &(p_real[state_index][1 - sense][ tile_width - 2 * halo_x]), tile_width * sizeof(double), halo_x * sizeof(double),1 );
        memcpy2D(&(p_imag[state_index][1 - sense][0]), tile_width * sizeof(double), &(p_imag[state_index][1 - sense][ tile_width - 2 * halo_x]), tile_width * sizeof(double), halo_x * sizeof(double),1 );
        memcpy2D(&(p_real[state_index][1 - sense][ tile_width - halo_x]), tile_width * sizeof(double), &(p_real[state_index][1 - sense][ halo_x]), tile_width * sizeof(double), halo_x * sizeof(double),1 );
        memcpy2D(&(p_imag[state_index][1 - sense][ tile_width - halo_x]), tile_width * sizeof(double), &(p_imag[state_index][1 - sense][ halo_x]), tile_width * sizeof(double), halo_x * sizeof(double),1 );
    }
#endif
}
