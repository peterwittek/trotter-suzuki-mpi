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

/**
 * This source provides an example of the trotter-suzuki program.
 * It calculates the time-evolution of a particle in a box, where the initial
 * state is the following:
 * 		exp(i2M_PI / L (x + y))
 */

#include <string>
#include <sstream>
#include <iostream>
#include <complex>
#include <sys/stat.h>

#include "common.h"
#include "trotter.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define LENGHT 50
#define DIM 640
#define ITERATIONS 2000
#define PARTICLES_NUM 1700000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 10
#define SCATTER_LENGHT_2D 5.662739242e-5

int rot_coord_x = 320, rot_coord_y = 320;
double omega = 0;

std::complex<double> gauss_ini_state(int m, int n, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
	double delta_x = double(LENGHT)/double(DIM);
    double x = (m - matrix_width / 2.) * delta_x, y = (n - matrix_height / 2.) * delta_x;
    double w = 0.01;
    return std::complex<double>(sqrt(w * double(PARTICLES_NUM) / M_PI) * exp(-(x * x + y * y) * 0.5 * w), 0.0);
}

double parabolic_potential(int m, int n, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    double delta_x = double(LENGHT)/double(DIM);
    double x = (m - matrix_width / 2.) * delta_x, y = (n - matrix_width / 2.) * delta_x;
    double w_x = 1, w_y = 1. / sqrt(2); 
    return 0.5 * (w_x * w_x * x * x + w_y * w_y * y * y);
}

int main(int argc, char** argv) {

    int dim = DIM, iterations = ITERATIONS, snapshots = SNAPSHOTS;
    string kernel_type = KERNEL_TYPE;
    int periods[2] = {0, 0};
    bool verbose = true;
    char filename[1] = "";
    char pot_name[1] = "";
    int halo_x = 4;
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;
    double norm = 1;
    bool imag_time = true;
    double delta_t = 1.e-4;
	const double particle_mass = 1.;
	
    //define the topology
    int coords[2], dims[2] = {0, 0};
    int rank;
    int nProcs;
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm cartcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Dims_create(nProcs, 2, dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords);
#else
    nProcs = 1;
    rank = 0;
    dims[0] = dims[1] = 1;
    coords[0] = coords[1] = 0;
#endif

    //set dimension of tiles and offsets
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    int tile_width = end_x - start_x;
    int tile_height = end_y - start_y;

    //set and calculate evolution operator variables from hamiltonian
    double coupling_const = 4. * M_PI * double(SCATTER_LENGHT_2D) * double(PARTICLES_NUM);
    double delta_x = double(LENGHT)/double(DIM), delta_y = double(LENGHT)/double(DIM);
    double *external_pot_real = new double[tile_width * tile_height];
    double *external_pot_imag = new double[tile_width * tile_height];
    double (*hamiltonian_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
    hamiltonian_pot = parabolic_potential;
    double time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
    double h_a = cosh(time_single_it / (2. * particle_mass * delta_x * delta_y));
    double h_b = sinh(time_single_it / (2. * particle_mass * delta_x * delta_y));

    initialize_exp_potential(external_pot_real, external_pot_imag, pot_name, hamiltonian_pot, tile_width, tile_height, matrix_width, matrix_height,
                             start_x, start_y, periods, coords, dims, halo_x, halo_y, time_single_it, particle_mass, imag_time);

    //set initial state
    double *p_real = new double[tile_width * tile_height];
    double *p_imag = new double[tile_width * tile_height];
    std::complex<double> (*ini_state)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
    ini_state = gauss_ini_state;
    initialize_state(p_real, p_imag, filename, ini_state, tile_width, tile_height, matrix_width, matrix_height, start_x, start_y,
                     periods, coords, dims, halo_x, halo_y);

    if(rank == 0) {
        std::cout << "\n* This source provides an example of the trotter-suzuki program.\n";
        std::cout << "* It calculates the time-evolution of a particle in a box\n";
        std::cout << "* with periodic boundary conditions, where the initial\n";
        std::cout << "* state is the following:\n";
        std::cout << "* \texp(i2M_PI / L (x + y))\n\n";
    }

    //set file output directory
    std::stringstream dirname;
    std::string dirnames;
    if(snapshots) {
        int status;

        dirname.str("");
        dirname << "D" << dim << "_I" << iterations << "_S" << snapshots << "";
        dirnames = dirname.str();

        status = mkdir(dirnames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if(status != 0 && status != -1)
            dirnames = ".";
    }
    else
        dirnames = ".";

    stamp(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, start_x, inner_start_x, inner_end_x, end_x,
          start_y, inner_start_y, inner_end_y, dims, coords, periods,
          0, iterations, 0, dirnames.c_str()
#ifdef HAVE_MPI
          , cartcomm
#endif
         );
    for(int count_snap = 1; count_snap <= snapshots; count_snap++) {
        trotter(h_a, h_b, coupling_const, external_pot_real, external_pot_imag, p_real, p_imag, delta_x, delta_y, matrix_width, matrix_height, delta_t, iterations, omega, rot_coord_x, rot_coord_y, kernel_type, norm, imag_time, periods);

        stamp(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, start_x, inner_start_x, inner_end_x, end_x,
              start_y, inner_start_y, inner_end_y, dims, coords, periods,
              0, iterations, count_snap, dirnames.c_str()
#ifdef HAVE_MPI
              , cartcomm
#endif
             );
    }
    if (coords[0] == 0 && coords[1] == 0 && verbose == true) {
        std::cout << "TROTTER " << matrix_width - periods[1] * 2 * halo_x << "x" << matrix_height - periods[0] * 2 * halo_y << " kernel:" << kernel_type << " np:" << nProcs << std::endl;
    }
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
