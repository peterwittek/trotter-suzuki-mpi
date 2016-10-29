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
#include <fstream>
#include <iostream>
#include "trottersuzuki1D.h"
#include "common.h"
#include<math.h>

void calculate_borders1D(int coord, int dim, int * start, int *end, int *inner_start, int *inner_end, int length, int halo, int periodic_bound) {
    int inner = (int)ceil((double)length / (double)dim);
    *inner_start = coord * inner;
    if(periodic_bound != 0)
        *start = *inner_start - halo;
    else
        *start = ( coord == 0 ? 0 : *inner_start - halo );
    *end = *inner_start + (inner + halo);

    if (*end > length) {
        if(periodic_bound != 0)
            *end = length + halo;
        else
            *end = length;
    }
    if(periodic_bound != 0)
        *inner_end = *end - halo;
    else
        *inner_end = ( *end == length ? *end : *end - halo );
}

double const_potential1D(double x ) {
    return 0.;
}

Lattice1D::Lattice1D(int dim, double _length,
                 bool periodic_x_axis): length_x(_length) {
    delta_x = length_x / double(dim);
    periods[0] = (int) periodic_x_axis;
    mpi_dims[0] = 0;
#ifdef HAVE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_procs);
    MPI_Dims_create(mpi_procs, 2, mpi_dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, mpi_dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &mpi_rank);
    MPI_Cart_coords(cartcomm, mpi_rank, 2, mpi_coords);
#else
    mpi_procs = 1;
    mpi_rank = 0;
    mpi_dims[0] = 1;
    mpi_coords[0] = 0;
#endif
  	halo_x=8;
    global_dim_x = dim + periods[0] * 2 * halo_x;
    global_no_halo_dim_x = dim;
    
    //set dimension of tiles and offsets
    calculate_borders1D(mpi_coords[0], mpi_dims[0], &start_x, &end_x,
                      &inner_start_x, &inner_end_x,
                      dim, halo_x, periods[0]);
    dim_x = end_x - start_x;
   
}



State1D::State1D(Lattice1D *_grid, double *_p_real, double *_p_imag): grid(_grid) {
    expected_values_updated = false;
    if (_p_real == 0) {
        self_init = true;
        p_real = new double[grid->dim_x ];
        for (int i = 0; i < grid->dim_x ; i++) {
            p_real[i] = 0;
        }
    }
    else {
        self_init = false;
        p_real = _p_real;
    }
    if (_p_imag == 0) {
        p_imag = new double[grid->dim_x ];
        for (int i = 0; i < grid->dim_x ; i++) {
            p_imag[i] = 0;
        }
    }
    else {
        p_imag = _p_imag;
    }
}

State1D::State1D(const State1D &obj): grid(obj.grid), expected_values_updated(obj.expected_values_updated), self_init(obj.self_init),
    mean_X(obj.mean_X), mean_XX(obj.mean_XX), 
    mean_Px(obj.mean_Px), mean_PxPx(obj.mean_PxPx), 
    norm2(obj.norm2) {
    p_real = new double[grid->dim_x ];
    p_imag = new double[grid->dim_x ];
    
        for (int x = 0; x < grid->dim_x; x++) {
            p_real[ x] = obj.p_real[ x];
            p_imag[ x] = obj.p_imag[ x];
        }
}

State1D::~State1D() {
    if (self_init) {
        delete [] p_real;
        delete [] p_imag;
    }
}

void State1D::imprint(complex<double> (*function)(double x)) {
    double delta_x = grid->delta_x ;
    double idx;
    double x_c = grid->global_no_halo_dim_x * grid->delta_x * 0.5;
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            complex<double> tmp = function(idx - x_c);
            double tmp_p_real = p_real[ grid->dim_x + x];
            p_real[ x] = tmp_p_real * real(tmp) - p_imag[ x] * imag(tmp);
            p_imag[ x] = tmp_p_real * imag(tmp) + p_imag[ x] * real(tmp);
        }
    //}
}

void State1D::init_state(complex<double> (*ini_state)(double x )) {
    complex<double> tmp;
    double delta_x = grid->delta_x;
    double idx;
    double x_c = grid->global_no_halo_dim_x * grid->delta_x * 0.5;
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = ini_state(idx - x_c);
            p_real[ x] = real(tmp);
            p_imag[ x] = imag(tmp);
        }
    //}
}

void State1D::loadtxt(char *file_name) {
    ifstream input(file_name);
    int in_width = grid->global_no_halo_dim_x;
    complex<double> tmp;
        for(int j = 0; j < in_width; j++) {
            input >> tmp;
            if((j - grid->start_x) >= 0 && (j - grid->start_x) < grid->dim_x) {
                p_real[ j - grid->start_x] = real(tmp);
                p_imag[ j - grid->start_x] = imag(tmp);
            }

            //Down band
                if((j - grid->start_x) >= 0 && (j - grid->start_x) < grid->dim_x) {
                    p_real[ j - grid->start_x] = real(tmp);
                    p_imag[ j - grid->start_x] = imag(tmp);
                }
                //Down right corner
                if(j < grid->halo_x && grid->periods[0] != 0 && grid->mpi_coords[0] == grid->mpi_dims[0] - 1) {
                    p_real[ j + grid->dim_x - grid->halo_x] = real(tmp);
                    p_imag[ j + grid->dim_x - grid->halo_x] = imag(tmp);
                }
                //Down left corner
                if(j >= in_width - grid->halo_x && grid->periods[0] != 0 && grid->mpi_coords[0] == 0) {
                    p_real[ j - (in_width - grid->halo_x)] = real(tmp);
                    p_imag[ j - (in_width - grid->halo_x)] = imag(tmp);
                }

            //Upper band
            
                if((j - grid->start_x) >= 0 && (j - grid->start_x) < grid->dim_x) {
                    p_real[ j - grid->start_x] = real(tmp);
                    p_imag[ j - grid->start_x] = imag(tmp);
                }
                //Up right corner
                if(j < grid->halo_x && grid->periods[0] != 0 && grid->mpi_coords[0] == grid->mpi_dims[0] - 1) {
                    p_real[ j + grid->dim_x - grid->halo_x] = real(tmp);
                    p_imag[ j + grid->dim_x - grid->halo_x] = imag(tmp);
                }
                //Up left corner
                if(j >= in_width - grid->halo_x && grid->periods[0] != 0 && grid->mpi_coords[0] == 0) {
                    p_real[ j - (in_width - grid->halo_x)] = real(tmp);
                    p_imag[ j - (in_width - grid->halo_x)] = imag(tmp);
                }

            //Right band
            if(j < grid->halo_x && grid->periods[0] != 0 && grid->mpi_coords[0] == grid->mpi_dims[0] - 1) {

                    p_real[ j + grid->dim_x - grid->halo_x] = real(tmp);
                    p_imag[ j + grid->dim_x - grid->halo_x] = imag(tmp);

            }

            //Left band
            if(j >= in_width - grid->halo_x && grid->periods[0] != 0 && grid->mpi_coords[0] == 0) {
                    p_real[ j - (in_width - grid->halo_x)] = real(tmp);
                    p_imag[ j - (in_width - grid->halo_x)] = imag(tmp);
            }
        }
    input.close();
}

double *State1D::get_particle_density(double *_density) {
    double *density;
    if (_density == 0) {
        density = new double[grid->dim_x];
    }
    else {
        density = _density;
    }
        for(int i = grid->inner_start_x - grid->start_x; i < grid->inner_end_x - grid->start_x; i++) {
            density[ i] = (p_real[ i] * p_real[ i] + p_imag[ i] * p_imag[ i]) * grid->delta_x;
        }
    return density;
}

void State1D::write_particle_density(string fileprefix) {
    double *density = new double[grid->dim_x];
    stringstream filename;
    filename << fileprefix << "-density";
    stamp_matrix1D(grid, get_particle_density(density), filename.str());
    delete [] density;
}

double *State1D::get_phase(double *_phase) {
    double *phase;
    if (_phase == 0) {
        phase = new double[grid->dim_x];
    }
    else {
        phase = _phase;
    }
    double norm;
        for(int i = grid->inner_start_x - grid->start_x; i < grid->inner_end_x - grid->start_x; i++) {
            norm = sqrt(p_real[ grid->dim_x + i] * p_real[ grid->dim_x + i] + p_imag[ grid->dim_x + i] * p_imag[ grid->dim_x + i]);
            if(norm == 0)
                phase[ i] = 0;
            else
                phase[ i] = acos(p_real[ i] / norm) * ((p_imag[ i] >= 0) - (p_imag[ i] < 0));
        }
    return phase;
}

void State1D::write_phase(string fileprefix) {
    double *phase = new double[grid->dim_x ];
    stringstream filename;
    filename << fileprefix << "-phase";
    stamp_matrix1D(grid, get_phase(phase), filename.str());
    delete [] phase;
}

void State1D::calculate_expected_values(void) {
    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int tile_width = grid->end_x - grid->start_x;
    double grid_center_x = grid->length_x * 0.5 - grid->delta_x * 0.5;
    double sum_norm2 = 0.;
    double sum_x_mean = 0, sum_xx_mean = 0;
    double sum_px_mean = 0, sum_pxpx_mean = 0, 
           param_px = - 1. / grid->delta_x;
    complex<double> const_1 = -1. / 12., const_2 = 4. / 3., const_3 = -2.5;
    complex<double> derivate1_1 = 1. / 6., derivate1_2 = - 1., derivate1_3 = 0.5, derivate1_4 = 1. / 3.;

#ifndef HAVE_MPI
    #pragma omp parallel for reduction(+:sum_norm2,sum_x_mean,sum_xx_mean,sum_px_mean,sum_pxpx_mean,)
#endif
        complex<double>  psi_center, psi_left, psi_right;
        complex<double>  psi_left_left, psi_right_right;
        int x = grid->inner_start_x;
        for (int j = grid->inner_start_x - grid->start_x; j < grid->inner_end_x - grid->start_x; ++j) {
            psi_center = complex<double> (p_real[ j], p_imag[ j]);
            sum_norm2 += real(conj(psi_center) * psi_center);
            sum_x_mean += real(conj(psi_center) * psi_center * complex<double>(grid->delta_x * x - grid_center_x, 0.));
            sum_xx_mean += real(conj(psi_center) * psi_center * complex<double>(grid->delta_x * x - grid_center_x, 0.) * complex<double>(grid->delta_x * x - grid_center_x, 0.));
            if (    j - (grid->inner_start_x - grid->start_x) >= (ini_halo_x == 0) * 2 &&
                    j < grid->inner_end_x - grid->start_x - (end_halo_x == 0) * 2) {
                psi_right = complex<double> (p_real[ tile_width + j + 1],
                                             p_imag[ tile_width + j + 1]);
                psi_left = complex<double> (p_real[ tile_width + j - 1],
                                            p_imag[ tile_width + j - 1]);
                psi_right_right = complex<double> (p_real[ tile_width + j + 2],
                                                   p_imag[ tile_width + j + 2]);
                psi_left_left = complex<double> (p_real[ tile_width + j - 2],
                                                 p_imag[ tile_width + j - 2]);

                sum_px_mean += imag(conj(psi_center) * (derivate1_4 * psi_right + derivate1_3 * psi_center + derivate1_2 * psi_left + derivate1_1 * psi_left_left));
                sum_pxpx_mean += real(conj(psi_center) * (const_1 * psi_right_right + const_2 * psi_right + const_2 * psi_left + const_1 * psi_left_left + const_3 * psi_center));

            }
            ++x;
        }
    norm2 = sum_norm2;
    mean_X = sum_x_mean;
    mean_XX = sum_xx_mean;
    mean_Px = - sum_px_mean * param_px;
    mean_PxPx = - sum_pxpx_mean * param_px * param_px;

#ifdef HAVE_MPI
    double *norm2_mpi = new double[grid->mpi_procs];
    double *mean_X_mpi = new double[grid->mpi_procs];
    double *mean_XX_mpi = new double[grid->mpi_procs];
    double *mean_Px_mpi = new double[grid->mpi_procs];
    double *mean_PxPx_mpi = new double[grid->mpi_procs];

    MPI_Allgather(&norm2, 1, MPI_DOUBLE, norm2_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_X, 1, MPI_DOUBLE, mean_X_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_XX, 1, MPI_DOUBLE, mean_XX_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_Px, 1, MPI_DOUBLE, mean_Px_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&mean_PxPx, 1, MPI_DOUBLE, mean_PxPx_mpi, 1, MPI_DOUBLE, grid->cartcomm);

    norm2 = 0.;
    mean_X = 0.;
    mean_XX = 0.;
    mean_Px = 0.;
    mean_PxPx = 0.;

    for(int i = 0; i < grid->mpi_procs; i++) {
        norm2 += norm2_mpi[i];
        mean_X += mean_X_mpi[i];
        mean_XX += mean_XX_mpi[i];
        mean_Px += mean_Px_mpi[i];
        mean_PxPx += mean_PxPx_mpi[i];
    }
    delete [] norm2_mpi;
    delete [] mean_X_mpi;
    delete [] mean_XX_mpi;
    delete [] mean_Px_mpi;
    delete [] mean_PxPx_mpi;
#endif
    mean_X = mean_X / norm2;
    mean_XX = mean_XX / norm2;
    mean_Px = mean_Px / norm2;
    mean_PxPx = mean_PxPx / norm2;
    norm2 *= grid->delta_x;
    expected_values_updated = true;
}

double State1D::get_mean_x(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_X;
}

double State1D::get_mean_xx(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_XX;
}


double State1D::get_mean_px(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_Px;
}

double State1D::get_mean_pxpx(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return mean_PxPx;
}


double State1D::get_squared_norm(void) {
    if(!expected_values_updated)
        calculate_expected_values();
    return norm2;
}

void State1D::write_to_file(string filename) {
    stamp1D(grid, this, filename);
}

ExponentialState1D::ExponentialState1D(Lattice1D *_grid, int _n_x, double _norm, double _phase, double *_p_real, double *_p_imag):
    State1D(_grid, _p_real, _p_imag), n_x(_n_x),  norm(_norm), phase(_phase) {
    complex<double> tmp;
    double delta_x = grid->delta_x; 
    double  idx;
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = exp_state(idx);
            p_real[ x] = real(tmp);
            p_imag[ x] = imag(tmp);
        }
}

complex<double> ExponentialState1D::exp_state(double x ) {
    double L_x = grid->global_no_halo_dim_x * grid->delta_x;
    return sqrt(norm / (L_x )) * exp(complex<double>(0., phase)) * exp(complex<double>(0., 2 * M_PI * double(n_x) / L_x * x));
}

GaussianState1D::GaussianState1D(Lattice1D *_grid, double _omega_x, double _mean_x,
                             double _norm, double _phase, double *_p_real, double *_p_imag):
    State1D(_grid, _p_real, _p_imag), mean_x(_mean_x),
     omega_x(_omega_x),  norm(_norm), phase(_phase) {

	complex<double> tmp;
    double delta_x = grid->delta_x;
    double  idx;
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = gauss_state(idx);
            p_real[ x] = real(tmp);
            p_imag[ x] = imag(tmp);
        }

}

complex<double> GaussianState1D::gauss_state(double x ) {
    double x_c = grid->global_no_halo_dim_x * grid->delta_x * 0.5;
    return complex<double>(sqrt(norm * sqrt(omega_x ) / M_PI) * exp(-(omega_x * pow(x - mean_x - x_c, 2.0) ) * 0.5), 0.) * exp(complex<double>(0., phase));
}

SinusoidState1D::SinusoidState1D(Lattice1D *_grid, int _n_x, double _norm, double _phase, double *_p_real, double *_p_imag):
    State1D(_grid, _p_real, _p_imag), n_x(_n_x), norm(_norm), phase(_phase)  {
    complex<double> tmp;
    double delta_x = grid->delta_x; 
    double  idx;
        idx = grid->start_x * delta_x + 0.5 * delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
            tmp = sinusoid_state(idx );
            p_real[ x] = real(tmp);
            p_imag[ x] = imag(tmp);
        }
}

complex<double> SinusoidState1D::sinusoid_state(double x) {
    double L_x = grid->global_no_halo_dim_x * grid->delta_x;
    return sqrt(norm / (L_x )) * 2.* exp(complex<double>(0., phase)) * complex<double> (sin(2 * M_PI * double(n_x) / L_x * x), 0.0);
}

Potential1D::Potential1D(Lattice1D *_grid, char *filename): grid(_grid) {
    matrix = new double[ grid->dim_x];
    self_init = true;
    is_static = true;
    ifstream input(filename);
    double tmp;
        for(int x = 0; x < grid->dim_x; x++) {
            input >> tmp;  
			matrix[ x] = tmp;
        }
    input.close();
}

Potential1D::Potential1D(Lattice1D *_grid, double *_external_pot): grid(_grid) {
    if (_external_pot == 0) {
        self_init = true;
        matrix = new double[grid->dim_x];
    }
    else {
        matrix = _external_pot;
    }
    self_init = false;
    is_static = true;
    evolving_potential = NULL;
    static_potential = NULL;
}

Potential1D::Potential1D(Lattice1D *_grid, double (*potential_fuction)(double x)): grid(_grid) {
    is_static = true;
    self_init = false;
    evolving_potential = NULL;
    static_potential = potential_fuction;
    matrix = NULL;
}

Potential1D::Potential1D(Lattice1D *_grid, double (*potential_function)(double x, double t), int _t): grid(_grid) {
    is_static = false;
    self_init = false;
    evolving_potential = potential_function;
    static_potential = NULL;
    matrix = NULL;
}

double Potential1D::get_value(int x) {
    if (matrix != NULL) {
        return matrix[ grid->dim_x + x];
    }
    else {
        double idx = grid->start_x * grid->delta_x + x * grid->delta_x + 0.5 * grid->delta_x;
        if (is_static) {
            return static_potential(idx);
        }
        else {
            return evolving_potential(idx, current_evolution_time);
        }
    }
}

bool Potential1D::update(double t) {
    current_evolution_time = t;
    if (!is_static) {
        if (matrix != NULL) {
            double delta_x = grid->delta_x;
            double idx;
                idx = grid->start_x * delta_x + 0.5 * delta_x;
                for (int x = 0; x < grid->dim_x; x++, idx += delta_x) {
                    matrix[ x] = evolving_potential(idx, t);
                }
        }
        return true;
    }
    return false;
}

Potential1D::~Potential1D() {
    if (self_init) {
        delete [] matrix;
    }
}

HarmonicPotential1D::HarmonicPotential1D(Lattice1D *_grid, double _omegax, double _mass, double _mean_x ):
    Potential1D(_grid, const_potential1D), omegax(_omegax), 
    mass(_mass), mean_x(_mean_x)  {
    is_static = true;
    self_init = false;
    evolving_potential = NULL;
    static_potential = NULL;
    matrix = NULL;
}

double HarmonicPotential1D::get_value(int x) {
    double idx = (grid->start_x + x) * grid->delta_x + mean_x + 0.5 * grid->delta_x;
    double x_c = (grid->global_dim_x - 2.*grid->halo_x * grid->periods[0]) * grid->delta_x * 0.5;
    double x_r = idx - x_c;
    return 0.5 * mass * (omegax * omegax * x_r * x_r);
}

HarmonicPotential1D::~HarmonicPotential1D() {
}

Hamiltonian1D::Hamiltonian1D(Lattice1D *_grid, Potential1D *_potential,
                         double _mass, double _coupling_a,
                         double _rot_coord_x): mass(_mass),
    coupling_a(_coupling_a), grid(_grid) {
        if (grid->periods[0] != 0 ) {
            cout << "Boundary conditions must be closed for rotating frame of refernce\n";
            return;
        }
        if (grid->mpi_procs == 1) {
            grid->halo_x = 8;
        }
        if (grid->mpi_procs > 1 && (grid->halo_x == 4)) {
            cout << "Halos must be of 8 points width\n";
            return;
        }
    rot_coord_x = (grid->global_dim_x - grid->periods[0] * 2 * grid->halo_x) * 0.5 + _rot_coord_x / grid->delta_x;
    if (_potential == NULL) {
        self_init = true;
        potential = new Potential1D(grid, const_potential1D);
    }
    else {
        self_init = false;
        potential = _potential;
    }
}

Hamiltonian1D::~Hamiltonian1D() {
    if (self_init) {
        delete potential;
    }
}

Hamiltonian2Component1D::Hamiltonian2Component1D(Lattice1D *_grid,
        Potential1D *_potential,
        Potential1D *_potential_b,
        double _mass,
        double _mass_b, double _coupling_a,
        double _coupling_ab, double _coupling_b,
        double _omega_r, double _omega_i,
        double _rot_coord_x):
    Hamiltonian1D(_grid, _potential, _mass, _coupling_a, _rot_coord_x ), mass_b(_mass_b),
    coupling_ab( _coupling_ab), coupling_b(_coupling_b), omega_r(_omega_r), omega_i(_omega_i) {

    if (_potential_b == NULL) {
        potential_b = _potential;
    }
    else {
        potential_b = _potential_b;
    }
}

Hamiltonian2Component1D::~Hamiltonian2Component1D() {

}
