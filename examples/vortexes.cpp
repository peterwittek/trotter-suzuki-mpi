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

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <complex>
#include <sys/stat.h>

#include "trotter.h"
#include "kernel.h"
#include "common.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#define LENGHT 50
#define DIM 640
#define ITERATIONS 10000
#define PARTICLES_NUM 8.e+6
#define KERNEL_TYPE "cpu"
#define SNAPSHOTS 400
#define SNAP_PER_STAMP 20
#define COUPLING_CONST_2D 7.116007999594e-4

int rot_coord_x = 320, rot_coord_y = 320;
double omega = 0.9;

std::complex<double> gauss_ini_state(int m, int n, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
	double delta_x = double(LENGHT)/double(DIM);
    double x = (m - matrix_width / 2.) * delta_x, y = (n - matrix_height / 2.) * delta_x;
    double w = 0.002/(delta_x * delta_x);
    return std::complex<double>(sqrt(0.5 * w * double(PARTICLES_NUM) / M_PI) * exp(-(x * x + y * y) * 0.5 * w), sqrt(0.5 * w * double(PARTICLES_NUM) / M_PI) * exp(-(x * x + y * y) * 0.5 * w));
}

std::complex<double> sinus_state(int m, int n, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
	double delta_x = double(LENGHT)/double(DIM);
	double x = m * delta_x, y = n * delta_x;
	return std::complex<double>(sin(2 * M_PI * x / double(LENGHT)) * sin(2 * M_PI * y / double(LENGHT)), 0.0);
}

double parabolic_potential(int m, int n, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    double delta_x = double(LENGHT)/double(DIM);
    double x = (m - matrix_width / 2.) * delta_x, y = (n - matrix_width / 2.) * delta_x;
    double w_x = 1., w_y = 1.; 
    return 0.5 * (w_x * w_x * x * x + w_y * w_y * y * y);
}

int main(int argc, char** argv) {
    int dim = DIM, iterations = ITERATIONS, snapshots = SNAPSHOTS, snap_per_stamp = SNAP_PER_STAMP;
    string kernel_type = KERNEL_TYPE;
    int periods[2] = {0, 0};
    char file_name[] = "";
    char pot_name[1] = "";
    const double particle_mass = 1.;
    bool imag_time = true;
    double h_a = 0.;
    double h_b = 0.;
	
	double delta_t = 2.e-4;
	double delta_x = double(LENGHT)/double(DIM), delta_y = double(LENGHT)/double(DIM);

    int halo_x = 4;
    halo_x = (omega == 0. ? halo_x : 8);
    int halo_y = (omega == 0. ? 4 : 8);
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
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
    
    //set and calculate evolution operator variables from hamiltonian
    double time_single_it;
    double coupling_const = double(COUPLING_CONST_2D);
    double *external_pot_real = new double[tile_width * tile_height];
    double *external_pot_imag = new double[tile_width * tile_height];
    double (*hamiltonian_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
    hamiltonian_pot = parabolic_potential;

    if(imag_time) {
        time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
        if(h_a == 0. && h_b == 0.) {
            h_a = cosh(time_single_it / (2. * particle_mass * delta_x * delta_y));
            h_b = sinh(time_single_it / (2. * particle_mass * delta_x * delta_y));
        }
    }
    else {
        time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
        if(h_a == 0. && h_b == 0.) {
            h_a = cos(time_single_it / (2. * particle_mass * delta_x * delta_y));
            h_b = sin(time_single_it / (2. * particle_mass * delta_x * delta_y));
        }
    }
    initialize_exp_potential(external_pot_real, external_pot_imag, pot_name, hamiltonian_pot, tile_width, tile_height, matrix_width, matrix_height,
                             start_x, start_y, periods, coords, dims, halo_x, halo_y, time_single_it, particle_mass, imag_time);

    //set initial state
    double *p_real = new double[tile_width * tile_height];
    double *p_imag = new double[tile_width * tile_height];
    std::complex<double> (*ini_state)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
    ini_state = gauss_ini_state;
    initialize_state(p_real, p_imag, file_name, ini_state, tile_width, tile_height, matrix_width, matrix_height, start_x, start_y,
                     periods, coords, dims, halo_x, halo_y);

    //set file output directory
    std::stringstream dirname, file_info;
    std::string dirnames, file_infos, file_tags;
    if(snapshots) {
        int status = 0;

        dirname.str("");
        dirname << "vortexesdir";
        dirnames = dirname.str();

        status = mkdir(dirnames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if(status != 0 && status != -1)
            dirnames = ".";
    }
    else
        dirnames = ".";
	
	file_info.str("");
	file_info << dirnames << "/file_info.txt";
	file_infos = file_info.str();
	std::ofstream out(file_infos.c_str());
	
	double *_matrix = new double[tile_width * tile_height];
    double *sums = new double[nProcs];
    double _kin_energy, _tot_energy, _rot_energy, _norm2, sum;
    
    double norm2;
	//norm calculation
	sum = Norm2(p_real, p_imag, delta_x, delta_y, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y);
#ifdef HAVE_MPI
	MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
	sums[0] = sum;
#endif 
	norm2 = 0.;
	for(int i = 0; i < nProcs; i++)
		norm2 += sums[i];
	
	//rot-energy calculation
	sum = Energy_rot(p_real, p_imag, omega, rot_coord_x, rot_coord_y, delta_x, delta_y, norm2, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y);
#ifdef HAVE_MPI
	MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
	sums[0] = sum;
#endif 
	_rot_energy = 0.;
	for(int i = 0; i < nProcs; i++)
		_rot_energy += sums[i];
	
	//tot-energy calculation
	sum = Energy_tot(p_real, p_imag, particle_mass, coupling_const, hamiltonian_pot, NULL, omega, rot_coord_x, rot_coord_y, delta_x, delta_y, norm2, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y, dim, dim, halo_x, halo_y, periods);
#ifdef HAVE_MPI
	MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
	sums[0] = sum;
#endif 
	_tot_energy = 0.;
	for(int i = 0; i < nProcs; i++)
		_tot_energy += sums[i];
	
	//kin energy
	sum = Energy_kin(p_real, p_imag, particle_mass, delta_x, delta_y, norm2, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y);
#ifdef HAVE_MPI
	MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
	sums[0] = sum;
#endif 
	_kin_energy = 0.;
	for(int i = 0; i < nProcs; i++)
		_kin_energy += sums[i];
			  
	if(rank == 0){
		out << "iterations \t rotation energy \t kin energy \t total energy \t norm2\n";
		out << "0\t" << _rot_energy << "\t" << _kin_energy << "\t" << _tot_energy << "\t" << norm2 << std::endl;
	}
		
    for(int count_snap = 0; count_snap < snapshots; count_snap++) {
        
        trotter(h_a, h_b, coupling_const, external_pot_real, external_pot_imag, p_real, p_imag, delta_x, delta_y, matrix_width, matrix_height, delta_t, iterations, omega, rot_coord_x, rot_coord_y, kernel_type, norm2, imag_time, periods);
      
        //norm calculation
        sum = Norm2(p_real, p_imag, delta_x, delta_y, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y);
#ifdef HAVE_MPI
        MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
        sums[0] = sum;
#endif 
        _norm2 = 0.;
        for(int i = 0; i < nProcs; i++)
            _norm2 += sums[i];
       
        //rot-energy calculation
        sum = Energy_rot(p_real, p_imag, omega, rot_coord_x, rot_coord_y, delta_x, delta_y, _norm2, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y);
#ifdef HAVE_MPI
        MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
        sums[0] = sum;
#endif 
		_rot_energy = 0.;
        for(int i = 0; i < nProcs; i++)
            _rot_energy += sums[i];
        
        //tot-energy calculation
        sum = Energy_tot(p_real, p_imag, particle_mass, coupling_const, hamiltonian_pot, NULL, omega, rot_coord_x, rot_coord_y, delta_x, delta_y, _norm2, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y, matrix_width, matrix_height, halo_x, halo_y, periods);
#ifdef HAVE_MPI
        MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
        sums[0] = sum;
#endif 
		_tot_energy = 0.;
        for(int i = 0; i < nProcs; i++)
            _tot_energy += sums[i];
        
        //kin energy
        sum = Energy_kin(p_real, p_imag, particle_mass, delta_x, delta_y, norm2, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y);
#ifdef HAVE_MPI
        MPI_Allgather(&sum, 1, MPI_DOUBLE, sums, 1, MPI_DOUBLE, cartcomm);
#else
        sums[0] = sum;
#endif 
		_kin_energy = 0.;
        for(int i = 0; i < nProcs; i++)
            _kin_energy += sums[i];
                  
        if(rank == 0){
			out << (count_snap + 1) * iterations << "\t" << _rot_energy << "\t" << _kin_energy << "\t" << _tot_energy << "\t" << _norm2 << std::endl;
		}
		
        //stamp phase and particles density
        if(count_snap % snap_per_stamp == 0.) {
			//get and stamp phase
			get_wave_function_phase(_matrix, p_real, p_imag, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y);
			file_tags = "phase";
			stamp_real(_matrix, matrix_width, matrix_height, halo_x, halo_y, start_x, inner_start_x, inner_end_x, end_x,
			   start_y, inner_start_y, inner_end_y, dims, coords, periods,
			   iterations * (count_snap + 1), dirnames.c_str(), file_tags.c_str()
#ifdef HAVE_MPI
			   , cartcomm
#endif
               );
        
			//get and stamp particles density
			get_wave_function_density(_matrix, p_real, p_imag, inner_start_x, start_x, inner_end_x, end_x, inner_start_y, start_y, inner_end_y, end_y);
			file_tags = "density";
			stamp_real(_matrix, matrix_width, matrix_height, halo_x, halo_y, start_x, inner_start_x, inner_end_x, end_x,
			   start_y, inner_start_y, inner_end_y, dims, coords, periods,
			   iterations * (count_snap + 1), dirnames.c_str(), file_tags.c_str()
#ifdef HAVE_MPI
				, cartcomm
#endif
				);
        }
    }
	
	out.close();
	stamp(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, start_x, inner_start_x, inner_end_x, end_x,
              start_y, inner_start_y, inner_end_y, dims, coords, periods,
              0, iterations, snapshots, dirnames.c_str()
#ifdef HAVE_MPI
              , cartcomm
#endif
             );
    

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
