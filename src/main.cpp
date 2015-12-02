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

#include <string.h>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <complex>

#ifdef WIN32
#include "unistd.h"
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

#if HAVE_CONFIG_H
#include "config.h"
#endif
#include "common.h"
#include "trotter.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#define DIM 640
#define EDGE_LENGHT 640
#define SINGLE_TIME_STEP 0.01
#define ITERATIONS 1000
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 1
#define PARTICLE_MASS 1
#define COUPLING_CONST 0
#define FILENAME_LENGTH 255

int rot_coord_x = 320, rot_coord_y = 320;
double omega = 0;

void print_usage() {
    std::cout << "Usage:\n" \
              "     trotter [OPTION] -n filename\n" \
              "Arguments:\n" \
              "     -m NUMBER     Particle mass (default: " << PARTICLE_MASS << ")\n"\
              "     -c NUMBER     Coupling constant of the self-interacting term (default: " << COUPLING_CONST << ")\n"\
              "     -d NUMBER     Matrix dimension (default: " << DIM << ")\n" \
              "     -l NUMBER     Physical dimension of the square lattice's edge (default: " << EDGE_LENGHT << ")\n" \
              "     -t NUMBER     Single time step (default: " << SINGLE_TIME_STEP << ")\n" \
              "     -i NUMBER     Number of iterations before a snapshot (default: " << ITERATIONS << ")\n" \
              "     -g            Imaginary time evolution to evolve towards the ground state\n" \
              "     -k STRING     Kernel type (cpu, gpu, or hybrid; default: " << KERNEL_TYPE << "): \n" \
              "     -s NUMBER     Snapshots are taken at every NUMBER of iterations.\n" \
              "                   Zero means no snapshots. Default: " << SNAPSHOTS << ".\n"\
              "     -n STRING     Name of file that defines the initial state.\n"\
              "     -p STRING     Name of file that stores the potential operator (in coordinate representation)\n";
}

void process_command_line(int argc, char** argv, int *dim, double *delta_x, double *delta_y, int *iterations, int *snapshots, string *kernel_type, char *filename, double *delta_t, double *coupling_const, double *particle_mass, char *pot_name, bool *imag_time) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;
    *delta_t = double(SINGLE_TIME_STEP);
    *coupling_const = double(COUPLING_CONST);
    *particle_mass = double(PARTICLE_MASS);

	double lenght = double(EDGE_LENGHT);
    int c;
    bool file_supplied = false;
    while ((c = getopt (argc, argv, "gd:hi:k:s:n:t:l:p:c:m:")) != -1) {
        switch (c) {
        case 'g':
            *imag_time = true;
            break;
        case 'd':
            *dim = atoi(optarg);
            if (*dim <= 0) {
                fprintf (stderr, "The argument of option -d should be a positive integer.\n");
                abort ();
            }
            break;
        case 'i':
            *iterations = atoi(optarg);
            if (*iterations <= 0) {
                fprintf (stderr, "The argument of option -i should be a positive integer.\n");
                abort ();
            }
            break;
        case 'h':
            print_usage();
            abort ();
            break;
        case 'k':
            *kernel_type = optarg;
            if (*kernel_type != "cpu" && *kernel_type != "gpu" && *kernel_type != "hybrid") {
                fprintf (stderr, "The argument of option -t should be cpu, gpu, or hybrid.");
                abort();
            }
            break;
        case 's':
            *snapshots = atoi(optarg);
            if (*snapshots <= 0) {
                fprintf (stderr, "The argument of option -s should be a positive integer.\n");
                abort ();
            }
            break;
        case 'n':
            for(size_t i = 0; i < strlen(optarg); i++)
                filename[i] = optarg[i];
            file_supplied = true;
            break;
        case 'c':
            *coupling_const = atoi(optarg);
            break;
        case 'm':
            *particle_mass = atoi(optarg);
            if (delta_t <= 0) {
                fprintf (stderr, "The argument of option -m should be a positive real number.\n");
                abort ();
            }
            break;
        case 't':
            *delta_t = atoi(optarg);
            if (delta_t <= 0) {
                fprintf (stderr, "The argument of option -t should be a positive real number.\n");
                abort ();
            }
            break;
        case 'p':
            for(size_t i = 0; i < strlen(optarg); i++)
                pot_name[i] = optarg[i];
            break;
        case 'l':
            lenght = atoi(optarg);
            if (lenght <= 0) {
                fprintf (stderr, "The argument of option -l should be a positive real number.\n");
                abort ();
            }
            break;
        case '?':
            if (optopt == 'd' || optopt == 'i' || optopt == 'k' || optopt == 's') {
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                print_usage();
                abort ();
            }
            else if (isprint (optopt)) {
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                print_usage();
                abort ();
            }
            else {
                fprintf (stderr, "Unknown option character `\\x%x'.\n",  optopt);
                print_usage();
                abort ();
            }
        default:
            abort ();
        }
    }
    if(!file_supplied) {
        fprintf (stderr, "Initial state file has not been supplied\n");
        print_usage();
        abort();
    }
    *delta_x = lenght / double(*dim);
    *delta_y = lenght / double(*dim);
}

int main(int argc, char** argv) {
    int dim = 0, iterations = 0, snapshots = 0;
    string kernel_type = KERNEL_TYPE;
    int periods[2] = {0, 0};
    double particle_mass = 1.;
    char filename[FILENAME_LENGTH] = "";
    char pot_name[FILENAME_LENGTH] = "";
    bool verbose = true, imag_time = false;
    double h_a = .0, h_b = .0;
    double norm = 1;
    int time, tot_time = 0;
    char output_folder[2] = {'.', '\0'};
	double delta_t = 0;
	double coupling_const = 0;
	double delta_x = 1, delta_y = 1;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    process_command_line(argc, argv, &dim, &delta_x, &delta_y, &iterations, &snapshots, &kernel_type, filename, &delta_t, &coupling_const, &particle_mass, pot_name, &imag_time);
	
    int halo_x = (kernel_type == "sse" ? 3 : 4);
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;

    //define the topology
    int coords[2], dims[2] = {0, 0};
    int rank;
    int nProcs;

#ifdef HAVE_MPI
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

    
	int read_offset = 0;

	//set and calculate evolution operator variables from hamiltonian
	double time_single_it;
	double *external_pot_real = new double[tile_width * tile_height];
	double *external_pot_imag = new double[tile_width * tile_height];
	double (*hamiltonian_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
	hamiltonian_pot = const_potential;

	if(imag_time) {
		double constant = 6.;
		time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
		if(h_a == 0. && h_b == 0.) {
			h_a = cosh(time_single_it / (2. * particle_mass)) / constant;
			h_b = sinh(time_single_it / (2. * particle_mass)) / constant;
		}
	}
	else {
		time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
		if(h_a == 0. && h_b == 0.) {
			h_a = cos(time_single_it / (2. * particle_mass));
			h_b = sin(time_single_it / (2. * particle_mass));
		}
	}
	initialize_exp_potential(external_pot_real, external_pot_imag, pot_name, hamiltonian_pot, tile_width, tile_height, matrix_width, matrix_height,
							 start_x, start_y, periods, coords, dims, halo_x, halo_y, time_single_it, particle_mass, imag_time);

	//set initial state
	double *p_real = new double[tile_width * tile_height];
	double *p_imag = new double[tile_width * tile_height];
	std::complex<double> (*ini_state)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
	ini_state = NULL;
	initialize_state(p_real, p_imag, filename, ini_state, tile_width, tile_height, matrix_width, matrix_height, start_x, start_y,
					 periods, coords, dims, halo_x, halo_y, read_offset);

	for(int count_snap = 0; count_snap <= snapshots; count_snap++) {
		stamp(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, start_x, inner_start_x, inner_end_x, end_x,
			  start_y, inner_start_y, inner_end_y, dims, coords, periods,
			  0, iterations, count_snap, output_folder
#ifdef HAVE_MPI
			  , cartcomm
#endif
			 );

		if(count_snap != snapshots) {
#ifdef WIN32
			SYSTEMTIME start;
			GetSystemTime(&start);
#else
			struct timeval start, end;
			gettimeofday(&start, NULL);
#endif
			trotter(h_a, h_b, coupling_const, external_pot_real, external_pot_imag, p_real, p_imag, delta_x, delta_y, matrix_width, matrix_height, delta_t, iterations, omega, rot_coord_x, rot_coord_y, kernel_type, norm, imag_time);
#ifdef WIN32
			SYSTEMTIME end;
			GetSystemTime(&end);
			time = (end.wMinute - start.wMinute) * 60000 + (end.wSecond - start.wSecond) * 1000 + (end.wMilliseconds - start.wMilliseconds);
#else
			gettimeofday(&end, NULL);
			time = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
#endif
			tot_time += time;
		}
	}

    if (coords[0] == 0 && coords[1] == 0 && verbose == true) {
        std::cout << "TROTTER " << matrix_width - periods[1] * 2 * halo_x << "x" << matrix_height - periods[0] * 2 * halo_y << " kernel:" << kernel_type << " np:" << nProcs << " " << tot_time << std::endl;
    }
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
