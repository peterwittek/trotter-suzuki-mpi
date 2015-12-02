/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2015 Luca Calderaro, 2012-2015 Peter Wittek,
 * 2010-2012 Carlos Bederián
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

#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#if HAVE_CONFIG_H
#include "config.h"
#endif
#include "common.h"
#include "trotter.h"
#include "cpublock.h"
#ifdef SSE
#include "cpublocksse.h"
#endif
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef CUDA
#include "cc2kernel.h"
#include "hybrid.h"
#endif

void trotter(double h_a, double h_b, double coupling_const,
             double * external_pot_real, double * external_pot_imag,
             double * p_real, double * p_imag, double delta_x, double delta_y,
             const int matrix_width, const int matrix_height, double delta_t,
             const int iterations, double omega, int rot_coord_x, int rot_coord_y,
             string kernel_type, double norm, bool imag_time, int *periods) {

    if(periods == NULL) {
        periods = new int [2];
        periods[0] = 0; periods[1] = 0;
	}
	
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;

    int coords[2], dims[2] = {0, 0};
    int rank;
    int nProcs;

#ifdef HAVE_MPI
    MPI_Comm cartcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Dims_create(nProcs, 2, dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords); //Determines process coords in cartesian topology given rank in group
#else
    nProcs = 1;
    rank = 0;
    dims[0] = dims[1] = 1;
    coords[0] = coords[1] = 0;
#endif

    int halo_x = (kernel_type == "sse" ? 3 : 4);
    halo_x = (omega == 0. ? halo_x : 8);
    int halo_y = (omega == 0. ? 4 : 8);
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    int width = end_x - start_x;
    int height = end_y - start_y;

    // Initialize kernel
    ITrotterKernel * kernel;
    if (kernel_type == "cpu") {
        kernel = new CPUBlock(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, coupling_const * delta_t, delta_x, delta_y, matrix_width, matrix_height, halo_x, halo_y, periods, norm, imag_time, omega * delta_t * delta_x / (2 * delta_y), omega * delta_t * delta_y / (2 * delta_x), rot_coord_x, rot_coord_y
#ifdef HAVE_MPI
                              , cartcomm
#endif
                             );
    } else if (kernel_type == "sse") {
#ifdef SSE
        kernel = new CPUBlockSSEKernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, delta_x, delta_y, matrix_width, matrix_height, halo_x, halo_y, periods, norm, imag_time
#ifdef HAVE_MPI
                                       , cartcomm
#endif
                                      );
#else
		if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "SSE kernel not was not compiled.\n";
        }
        abort();
#endif
    } else if (kernel_type == "gpu") {
#ifdef CUDA
        kernel = new CC2Kernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, delta_x, delta_y, matrix_width, matrix_height, halo_x, halo_y, periods, norm, imag_time
#ifdef HAVE_MPI
                               , cartcomm
#endif
                              );
#else
        if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "Compiled without CUDA\n";
        }
#ifdef HAVE_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#else
        abort ();
#endif
#endif
    } else if (kernel_type == "hybrid") {
#ifdef CUDA
        kernel = new HybridKernel(p_real, p_imag, external_pot_real, external_pot_imag, h_a, h_b, coupling_const, delta_x, delta_y, matrix_width, matrix_height, halo_x, halo_y, periods, norm, imag_time
#ifdef HAVE_MPI
                                  , cartcomm
#endif
                                 );
#else
        if (coords[0] == 0 && coords[1] == 0) {
            std::cerr << "Compiled without CUDA\n";
        }
#ifdef HAVE_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#else
        abort();
#endif
#endif
    }

    // Main loop
    for (int i = 0; i < iterations; i++) {
        kernel->run_kernel_on_halo();
        if (i != iterations - 1) {
            kernel->start_halo_exchange();
        }
        kernel->run_kernel();
        if (i != iterations - 1) {
            kernel->finish_halo_exchange();
        }
        kernel->wait_for_completion(i);
    }

    kernel->get_sample(width, 0, 0, width, height, p_real, p_imag);

    delete kernel;
}
