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
#include "trottersuzuki.h"
#include "common.h"
#include "kernel.h"
#include <iostream>
#include <cstring>


Solver::Solver(Lattice *_grid, State *_state, Hamiltonian *_hamiltonian,
               double _delta_t, string _kernel_type):
    grid(_grid), state(_state), hamiltonian(_hamiltonian), delta_t(_delta_t),
    kernel_type(_kernel_type) {
    external_pot_real = new double* [2];
    external_pot_imag = new double* [2];
    external_pot_real[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_imag[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_real[1] = NULL;
    external_pot_imag[1] = NULL;
    is_python = false;
    state_b = NULL;
    kernel = NULL;
    current_evolution_time = 0;
    single_component = true;
    energy_expected_values_updated = false;
    has_parameters_changed = false;
}

Solver::Solver(Lattice *_grid, State *state1, State *state2,
               Hamiltonian2Component *_hamiltonian,
               double _delta_t, string _kernel_type):
    grid(_grid), state(state1), state_b(state2), hamiltonian(_hamiltonian), delta_t(_delta_t),
    kernel_type(_kernel_type) {
    external_pot_real = new double* [2];
    external_pot_imag = new double* [2];
    external_pot_real[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_imag[0] = new double[grid->dim_x * grid->dim_y];
    external_pot_real[1] = new double[grid->dim_x * grid->dim_y];
    external_pot_imag[1] = new double[grid->dim_x * grid->dim_y];
    is_python = false;
    kernel = NULL;
    current_evolution_time = 0;
    single_component = false;
    energy_expected_values_updated = false;
    has_parameters_changed = false;
}

Solver::~Solver() {
    delete [] external_pot_real[0];
    delete [] external_pot_imag[0];
    delete [] external_pot_real[1];
    delete [] external_pot_imag[1];
    delete [] external_pot_real;
    delete [] external_pot_imag;
    if (kernel != NULL) {
        delete kernel;
    }
}

void Solver::initialize_exp_potential(double delta_t, int which) {
#ifndef HAVE_MPI
    #pragma omp parallel default(shared)
#endif
    {
        complex<double> tmp;
        double ptmp;
#ifndef HAVE_MPI
        #pragma omp for schedule(dynamic, 12) collapse(2)
#endif
        for (int y = 0; y < grid->dim_y; ++y) {
            for (int x = 0; x < grid->dim_x; ++x) {
                if (which == 0) {
                    ptmp = hamiltonian->potential->get_value(x, y);
                    if (grid->coordinate_system == "cylindrical") {
                        ptmp += hamiltonian->azimuthal_potential(x, state->angular_momentum);
                    }
                }
                else {
                    ptmp = static_cast<Hamiltonian2Component*>(hamiltonian)->potential_b->get_value(x, y);
                    if (grid->coordinate_system == "cylindrical") {
                        ptmp += static_cast<Hamiltonian2Component*>(hamiltonian)->azimuthal_potential_b(x, state_b->angular_momentum);
                    }
                }
                if (imag_time) {
                    tmp = exp(complex<double> (-delta_t * ptmp, 0.));
                }
                else {
                    tmp = exp(complex<double> (0., -delta_t * ptmp));
                }
                external_pot_real[which][y * grid->dim_x + x] = real(tmp);
                external_pot_imag[which][y * grid->dim_x + x] = imag(tmp);
            }
        }
    }
}

void Solver::set_exp_potential(double *real, int real_length, double *imag,
                               int imag_length, int which) {
    is_python = true;
    memcpy(external_pot_real[which], real, sizeof(double)*real_length);
    memcpy(external_pot_imag[which], imag, sizeof(double)*imag_length);
}

void Solver::init_kernel() {
    if (kernel != NULL) {
        delete kernel;
    }
    if (kernel_type == "cpu") {
        if (single_component) {
            kernel = new CPUBlock(grid, state, hamiltonian, external_pot_real[0], external_pot_imag[0], delta_t, norm2[0], imag_time);
        }
        else {
            kernel = new CPUBlock(grid, state, state_b, static_cast<Hamiltonian2Component*>(hamiltonian), external_pot_real, external_pot_imag, delta_t, norm2, imag_time);
        }
    }
    else if (kernel_type == "gpu") {
#ifdef CUDA
        if (hamiltonian->angular_velocity != 0) {
            my_abort("The GPU kernel does not work with nonzero angular velocity.");
        }
        if (single_component) {
            kernel = new CC2Kernel(grid, state, hamiltonian, external_pot_real[0], external_pot_imag[0], delta_t, norm2[0], imag_time);
        }
        else {
            kernel = new CC2Kernel(grid, state, state_b, static_cast<Hamiltonian2Component*>(hamiltonian), external_pot_real, external_pot_imag, delta_t, norm2, imag_time);
        }
#else
        my_abort("Compiled without CUDA");
#endif
    }
    else {
        my_abort("Unknown kernel");
    }
}

void Solver::evolve(int iterations, bool _imag_time) {
    if (_imag_time != imag_time || kernel == NULL || has_parameters_changed) {
        imag_time = _imag_time;
        if (imag_time) {
            initialize_exp_potential(delta_t, 0);
            norm2[0] = state->get_squared_norm();
            if (!single_component) {
                initialize_exp_potential(delta_t, 1);
                norm2[1] = state_b->get_squared_norm();
            }
        }
        else {
            if (!is_python) {
                initialize_exp_potential(delta_t, 0);
            }
            if (!single_component) {
                initialize_exp_potential(delta_t, 1);
            }
        }
        init_kernel();
        has_parameters_changed = false;
    }
    // Main loop
    double var = 0.5;
    if ((!is_python && !single_component) ||
            (is_python && current_evolution_time == 0 && !single_component)) {
        kernel->rabi_coupling(var, delta_t);
    }
    var = 1.;
    bool soft_update = false;
    if (iterations < 0) {
        iterations = -iterations;
        soft_update = true;
    }

    // Main loop
    for (int i = 0; i < iterations; ++i) {
        if (i > 0 && hamiltonian->potential->update(current_evolution_time)) {
            if (!is_python) {
                initialize_exp_potential(delta_t, 0);
            }
            kernel->update_potential(external_pot_real[0], external_pot_imag[0], 0);
        }
        if (!single_component && i > 0) {
            if (static_cast<Hamiltonian2Component*>(hamiltonian)->potential_b->update(current_evolution_time)) {
                if (!is_python) {
                    initialize_exp_potential(delta_t, 1);
                }
                kernel->update_potential(external_pot_real[1], external_pot_imag[1], 1);
            }
        }
        //first wave function
        kernel->run_kernel_on_halo();
        if (i != iterations - 1) {
            kernel->start_halo_exchange();
        }
        kernel->run_kernel();
        if (i != iterations - 1) {
            kernel->finish_halo_exchange();
        }
        kernel->wait_for_completion();
        if (!single_component) {
            //second wave function
            kernel->run_kernel_on_halo();
            if (i != iterations - 1) {
                kernel->start_halo_exchange();
            }
            kernel->run_kernel();
            if (i != iterations - 1) {
                kernel->finish_halo_exchange();
            }
            kernel->wait_for_completion();
            if (i == iterations - 1) {
                var = 0.5;
            }
            kernel->rabi_coupling(var, delta_t);
            kernel->normalization();
        }
        kernel->cpy_first_positive_to_first_negative(); //only for cylindrical coordinates
        current_evolution_time += delta_t;
    }
    if (!soft_update) {
        if (single_component) {
            kernel->get_sample(grid->dim_x, 0, 0, grid->dim_x, grid->dim_y, state->p_real, state->p_imag);
        }
        else {
            kernel->get_sample(grid->dim_x, 0, 0, grid->dim_x, grid->dim_y, state->p_real, state->p_imag, state_b->p_real, state_b->p_imag);
            state_b->expected_values_updated = false;
        }
    }
    state->expected_values_updated = false;
    energy_expected_values_updated = false;
}

void Solver::calculate_energy_expected_values(void) {

    double delta_x = grid->delta_x;
    double delta_y = grid->delta_y;

    double sum_norm2_0 = 0;
    double sum_norm2_kin0 = 0;
    double sum_norm2_1 = 0;
    double sum_norm2_kin1 = 0;
    double sum_kinetic_energy_0 = 0;
    double sum_kinetic_energy_1 = 0;
    double sum_potential_energy_0 = 0;
    double sum_potential_energy_1 = 0;
    double sum_rotational_energy_0 = 0;
    double sum_rotational_energy_1 = 0;
    double sum_intra_species_energy_0 = 0;
    double sum_intra_species_energy_1 = 0;
    double sum_inter_species_energy = 0;
    double sum_rabi_energy = 0;
    double sum_LeeHuangYang = 0;

    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int ini_halo_y = grid->inner_start_y - grid->start_y;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int end_halo_y = grid->end_y - grid->inner_end_y;
    int tile_width = grid->end_x - grid->start_x;

    double x, y;
    Potential *potential, *potential_b = NULL;
    double coupling, coupling_b = 0, coupling_ab, LeeHuangYang_coupling_a;
    double mass, mass_b;
    double angular_velocity = hamiltonian->angular_velocity;
    complex<double> omega;

    potential = hamiltonian->potential;
    coupling = hamiltonian->coupling_a;
    LeeHuangYang_coupling_a = hamiltonian->LeeHuangYang_coupling_a;
    mass = hamiltonian->mass;
    if (!single_component) {
        potential_b = static_cast<Hamiltonian2Component*>(hamiltonian)->potential_b;
        coupling_b = static_cast<Hamiltonian2Component*>(hamiltonian)->coupling_b;
        coupling_ab = static_cast<Hamiltonian2Component*>(hamiltonian)->coupling_ab;
        mass_b = static_cast<Hamiltonian2Component*>(hamiltonian)->mass_b;
        omega = complex<double> (static_cast<Hamiltonian2Component*>(hamiltonian)->omega_r,
                                 static_cast<Hamiltonian2Component*>(hamiltonian)->omega_i);
    }

    double cost_E = -1. / (2. * mass);
    double cost_E_b;
    if (!single_component)
        cost_E_b = -1. / (2. * mass_b);

    // double cost_rot_x = angular_velocity * delta_y / delta_x;
    // double cost_rot_y = angular_velocity * delta_x / delta_y;

    complex<double> const_1 = -1. / 12., const_2 = 4. / 3., const_3 = -2.5;
    complex<double> derivate1_1 = 1. / 6., derivate1_2 = - 1., derivate1_3 = 0.5, derivate1_4 = 1. / 3.;
    int xlim = (grid->coordinate_system == "cylindrical" ? 3 : 0);

#ifndef HAVE_MPI
    #pragma omp parallel for reduction(+:sum_norm2_0,\
    sum_norm2_kin0,\
    sum_potential_energy_0,\
    sum_intra_species_energy_0,\
    sum_kinetic_energy_0,\
    sum_rotational_energy_0,\
    sum_LeeHuangYang,\
    sum_inter_species_energy,\
    sum_rabi_energy,\
    sum_norm2_1,\
    sum_norm2_kin1,\
    sum_potential_energy_1,\
    sum_intra_species_energy_1,\
    sum_kinetic_energy_1,\
    sum_rotational_energy_1) private(x,y)
#endif
    for (int i = grid->inner_start_y - grid->start_y; i < grid->inner_end_y - grid->start_y; ++i) {
    complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;
    complex<double> psi_up_b, psi_down_b, psi_center_b, psi_left_b, psi_right_b;
    complex<double> rot_y, rot_x;
    complex<double> psi_up_up, psi_down_down, psi_left_left, psi_right_right;
    complex<double> psi_up_up_b, psi_down_down_b, psi_left_left_b, psi_right_right_b;

    for (int j = grid->inner_start_x - grid->start_x; j < grid->inner_end_x - grid->start_x; ++j) {
            psi_center = complex<double> (state->p_real[i * tile_width + j],
                                          state->p_imag[i * tile_width + j]);
            map_lattice_to_coordinate_space(grid, j, i, &x, &y);
            complex<double> x_r = x;
            complex<double> y_r = y;

            double norm2 = real(conj(psi_center) * psi_center);
            sum_norm2_0 += norm2;
            sum_potential_energy_0 += norm2 * (potential->get_value(j, i) + (grid->coordinate_system == "cylindrical" ? hamiltonian->azimuthal_potential(j, state->angular_momentum) : 0));
            sum_intra_species_energy_0 += norm2 * norm2 * 0.5 * coupling;
            sum_LeeHuangYang += pow(norm2, 2.5) * 0.4 * LeeHuangYang_coupling_a;

            if (!single_component) {
                psi_center_b = complex<double> (state_b->p_real[i * tile_width + j],
                                                state_b->p_imag[i * tile_width + j]);

                sum_norm2_1 += real(conj(psi_center_b) * psi_center_b);
                sum_potential_energy_1 += real(conj(psi_center_b) * psi_center_b * complex<double> (
                                                   potential_b->get_value(j, i) +
                                                   (grid->coordinate_system == "cylindrical" ? static_cast<Hamiltonian2Component*>(hamiltonian)->azimuthal_potential_b(j, state_b->angular_momentum) : 0.), 0.));

                sum_intra_species_energy_1 += real(conj(psi_center_b) * psi_center_b * psi_center_b * conj(psi_center_b) * complex<double> (0.5 * coupling_b, 0.));
                sum_inter_species_energy += real(conj(psi_center) * psi_center * conj(psi_center) * psi_center *
                                                 conj(psi_center_b) * psi_center_b * conj(psi_center_b) * psi_center_b * complex<double> (coupling_ab));
                sum_rabi_energy += real(conj(psi_center) * psi_center * conj(psi_center_b) * psi_center_b * (conj(psi_center) * psi_center_b * omega +
                                        conj(psi_center_b * omega) * psi_center));
            }

            if (((i - (grid->inner_start_y - grid->start_y) >= (ini_halo_y == 0) * 2 &&
                    i < grid->inner_end_y - grid->start_y - (end_halo_y == 0) * 2) || grid->dim_y == 1) &&
                    (j - (grid->inner_start_x - grid->start_x) >= (ini_halo_x == 0) * 2 + xlim &&
                     j < grid->inner_end_x - grid->start_x - (end_halo_x == 0) * 2)) {

                psi_right = complex<double> (state->p_real[i * tile_width + j + 1],
                                             state->p_imag[i * tile_width + j + 1]);
                psi_left = complex<double> (state->p_real[i * tile_width + j - 1],
                                            state->p_imag[i * tile_width + j - 1]);
                psi_right_right = complex<double> (state->p_real[i * tile_width + j + 2],
                                                   state->p_imag[i * tile_width + j + 2]);
                psi_left_left = complex<double> (state->p_real[i * tile_width + j - 2],
                                                 state->p_imag[i * tile_width + j - 2]);
                psi_up = complex<double> (state->p_real[(i - 1) * tile_width + j],
                                          state->p_imag[(i - 1) * tile_width + j]);
                psi_down = complex<double> (state->p_real[(i + 1) * tile_width + j],
                                            state->p_imag[(i + 1) * tile_width + j]);
                psi_up_up = complex<double> (state->p_real[(i - 2) * tile_width + j],
                                             state->p_imag[(i - 2) * tile_width + j]);
                psi_down_down = complex<double> (state->p_real[(i + 2) * tile_width + j],
                                                 state->p_imag[(i + 2) * tile_width + j]);

                sum_norm2_kin0 += real(conj(psi_center) * psi_center);
                sum_kinetic_energy_0 += real(cost_E * conj(psi_center) *
                                             (const_1 * psi_right_right + const_2 * psi_right + const_2 * psi_left + const_1 * psi_left_left + const_3 * psi_center) / (delta_x * delta_x));
                if (grid->dim_y > 1) {
                    sum_kinetic_energy_0 += real(cost_E * conj(psi_center) *
                                                 (const_1 * psi_down_down + const_2 * psi_down + const_2 * psi_up + const_1 * psi_up_up + const_3 * psi_center) / (delta_y * delta_y));
                    rot_x = y_r * angular_velocity / delta_x * complex<double>(0., 1.);
                    rot_y = x_r * angular_velocity / delta_y * complex<double>(0., 1.);
                    sum_rotational_energy_0 += real(conj(psi_center) * (rot_y * (derivate1_4 * psi_up + derivate1_3 * psi_center + derivate1_2 * psi_down + derivate1_1 * psi_down_down)
                                                    + rot_x * (derivate1_4 * psi_right + derivate1_3 * psi_center + derivate1_2 * psi_left + derivate1_1 * psi_left_left)));
                }

                if (!single_component) {
                    psi_up_b = complex<double> (state_b->p_real[(i - 1) * tile_width + j],
                                                state_b->p_imag[(i - 1) * tile_width + j]);
                    psi_down_b = complex<double> (state_b->p_real[(i + 1) * tile_width + j],
                                                  state_b->p_imag[(i + 1) * tile_width + j]);
                    psi_right_b = complex<double> (state_b->p_real[i * tile_width + j + 1],
                                                   state_b->p_imag[i * tile_width + j + 1]);
                    psi_left_b = complex<double> (state_b->p_real[i * tile_width + j - 1],
                                                  state_b->p_imag[i * tile_width + j - 1]);
                    psi_up_up_b = complex<double> (state_b->p_real[(i - 2) * tile_width + j],
                                                   state_b->p_imag[(i - 2) * tile_width + j]);
                    psi_down_down_b = complex<double> (state_b->p_real[(i + 2) * tile_width + j],
                                                       state_b->p_imag[(i + 2) * tile_width + j]);
                    psi_right_right_b = complex<double> (state_b->p_real[i * tile_width + j + 2],
                                                         state_b->p_imag[i * tile_width + j + 2]);
                    psi_left_left_b = complex<double> (state_b->p_real[i * tile_width + j - 2],
                                                       state_b->p_imag[i * tile_width + j - 2]);

                    sum_norm2_kin1 += real(conj(psi_center_b) * psi_center_b);
                    sum_kinetic_energy_1 += real(cost_E_b * conj(psi_center_b) *
                                                 (const_1 * psi_right_right_b + const_2 * psi_right_b + const_2 * psi_left_b + const_1 * psi_left_left_b + const_3 * psi_center_b) / (delta_x * delta_x));
                    if (grid->dim_y > 1) {
                        sum_kinetic_energy_1 += real(cost_E_b * conj(psi_center_b) *
                                                     (const_1 * psi_down_down_b + const_2 * psi_down_b + const_2 * psi_up_b + const_1 * psi_up_up_b + const_3 * psi_center_b) / (delta_y * delta_y));
                        sum_rotational_energy_1 += real(conj(psi_center_b) * (rot_y * (derivate1_4 * psi_up_b + derivate1_3 * psi_center_b + derivate1_2 * psi_down_b + derivate1_1 * psi_down_down_b)
                                                        + rot_x * (derivate1_4 * psi_right_b + derivate1_3 * psi_center_b + derivate1_2 * psi_left_b + derivate1_1 * psi_left_left_b)));
                    }
                }
            }
        }
    }

    double *norm2_kin = new double[2];
    norm2_kin[0] = sum_norm2_kin0;
    norm2[0] = sum_norm2_0;
    kinetic_energy[0] = sum_kinetic_energy_0;
    potential_energy[0] = sum_potential_energy_0;
    rotational_energy[0] = sum_rotational_energy_0;
    intra_species_energy[0] = sum_intra_species_energy_0;
    LeeHuangYang_energy = sum_LeeHuangYang;

    if (!single_component) {
    norm2_kin[1] = sum_norm2_kin1;
        norm2[1] = sum_norm2_1;
        kinetic_energy[1] = sum_kinetic_energy_1;
        potential_energy[1] = sum_potential_energy_1;
        rotational_energy[1] = sum_rotational_energy_1;
        intra_species_energy[1] = sum_intra_species_energy_1;
        inter_species_energy = sum_inter_species_energy;
        rabi_energy = 0.5 * sum_rabi_energy;
    }

#ifdef HAVE_MPI
    double *norm2_kin_mpi = new double[grid->mpi_procs];
    double *norm2_mpi = new double[grid->mpi_procs];
    double *kinetic_energy_mpi = new double[grid->mpi_procs];
    double *potential_energy_mpi = new double[grid->mpi_procs];
    double *rotational_energy_mpi = new double[grid->mpi_procs];
    double *intra_species_energy_mpi = new double[grid->mpi_procs];
    double *LeeHuangYang_energy_mpi = new double[grid->mpi_procs];

    MPI_Allgather(&norm2_kin[0], 1, MPI_DOUBLE, norm2_kin_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&norm2[0], 1, MPI_DOUBLE, norm2_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&kinetic_energy[0], 1, MPI_DOUBLE, kinetic_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&potential_energy[0], 1, MPI_DOUBLE, potential_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&rotational_energy[0], 1, MPI_DOUBLE, rotational_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&intra_species_energy[0], 1, MPI_DOUBLE, intra_species_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&LeeHuangYang_energy, 1, MPI_DOUBLE, LeeHuangYang_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);

    norm2_kin[0] = 0;
    norm2[0] = 0;
    kinetic_energy[0] = 0;
    potential_energy[0] = 0;
    rotational_energy[0] = 0;
    intra_species_energy[0] = 0;
    LeeHuangYang_energy = 0;

    for(int i = 0; i < grid->mpi_procs; i++) {
    norm2_kin[0] += norm2_kin_mpi[i];
        norm2[0] += norm2_mpi[i];
        kinetic_energy[0] += kinetic_energy_mpi[i];
        potential_energy[0] += potential_energy_mpi[i];
        rotational_energy[0] += rotational_energy_mpi[i];
        intra_species_energy[0] += intra_species_energy_mpi[i];
        LeeHuangYang_energy += LeeHuangYang_energy_mpi[i];
    }
    delete [] norm2_kin_mpi;
    delete [] norm2_mpi;
    delete [] kinetic_energy_mpi;
    delete [] potential_energy_mpi;
    delete [] rotational_energy_mpi;
    delete [] intra_species_energy_mpi;
    delete [] LeeHuangYang_energy_mpi;

    if (!single_component) {
    double *norm2_kin_mpi = new double[grid->mpi_procs];
        double *norm2_mpi = new double[grid->mpi_procs];
        double *kinetic_energy_mpi = new double[grid->mpi_procs];
        double *potential_energy_mpi = new double[grid->mpi_procs];
        double *rotational_energy_mpi = new double[grid->mpi_procs];
        double *intra_species_energy_mpi = new double[grid->mpi_procs];
        double *inter_species_energy_mpi = new double[grid->mpi_procs];
        double *rabi_energy_mpi = new double[grid->mpi_procs];

        MPI_Allgather(&norm2_kin[1], 1, MPI_DOUBLE, norm2_kin_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&norm2[1], 1, MPI_DOUBLE, norm2_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&kinetic_energy[1], 1, MPI_DOUBLE, kinetic_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&potential_energy[1], 1, MPI_DOUBLE, potential_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&rotational_energy[1], 1, MPI_DOUBLE, rotational_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&intra_species_energy[1], 1, MPI_DOUBLE, intra_species_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&inter_species_energy, 1, MPI_DOUBLE, inter_species_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&rabi_energy, 1, MPI_DOUBLE, rabi_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);

        norm2_kin[1] = 0;
        norm2[1] = 0;
        kinetic_energy[1] = 0;
        potential_energy[1] = 0;
        rotational_energy[1] = 0;
        intra_species_energy[1] = 0;
        inter_species_energy = 0;
        rabi_energy = 0;

        for(int i = 0; i < grid->mpi_procs; i++) {
            norm2_kin[1] += norm2_kin_mpi[i];
            norm2[1] += norm2_mpi[i];
            kinetic_energy[1] += kinetic_energy_mpi[i];
            potential_energy[1] += potential_energy_mpi[i];
            rotational_energy[1] += rotational_energy_mpi[i];
            intra_species_energy[1] += intra_species_energy_mpi[i];
            inter_species_energy += inter_species_energy_mpi[i];
            rabi_energy += rabi_energy_mpi[i];
        }
        delete [] norm2_kin_mpi;
        delete [] norm2_mpi;
        delete [] kinetic_energy_mpi;
        delete [] potential_energy_mpi;
        delete [] rotational_energy_mpi;
        delete [] intra_species_energy_mpi;
        delete [] inter_species_energy_mpi;
        delete [] rabi_energy_mpi;
    }
#endif
    kinetic_energy[0] = kinetic_energy[0] / norm2_kin[0];
    rotational_energy[0] = rotational_energy[0] / norm2_kin[0];
    potential_energy[0] = potential_energy[0] / norm2[0];
    intra_species_energy[0] = intra_species_energy[0] / norm2[0];
    LeeHuangYang_energy = LeeHuangYang_energy / norm2[0];
    if (single_component) {
    total_energy = kinetic_energy[0] + potential_energy[0] + intra_species_energy[0] + rotational_energy[0] + LeeHuangYang_energy;
        tot_kinetic_energy = kinetic_energy[0];
        tot_potential_energy = potential_energy[0];
        tot_rotational_energy = rotational_energy[0];
        tot_intra_species_energy = intra_species_energy[0];
    }
    else {
        kinetic_energy[1] = kinetic_energy[1] / norm2_kin[1];
        rotational_energy[1] = rotational_energy[1] / norm2_kin[1];
        potential_energy[1] = potential_energy[1] / norm2[1];
        intra_species_energy[1] = intra_species_energy[1] / norm2[1];
        inter_species_energy = inter_species_energy / (norm2[0] * norm2[1]);
        rabi_energy = rabi_energy / (norm2[0] * norm2[1]);

        total_energy = kinetic_energy[0] + potential_energy[0] + intra_species_energy[0] + rotational_energy[0] +
                       kinetic_energy[1] + potential_energy[1] + intra_species_energy[1] + rotational_energy[1] +
                       inter_species_energy + rabi_energy;
        tot_kinetic_energy = kinetic_energy[0] + kinetic_energy[1];
        tot_potential_energy = potential_energy[0] + potential_energy[1];
        tot_rotational_energy = rotational_energy[0] + rotational_energy[1];
        tot_intra_species_energy = intra_species_energy[0] + intra_species_energy[1];
        norm2[1] *= delta_y * grid->length_x / (grid->global_no_halo_dim_x - (grid->coordinate_system == "cylindrical" ? 1 : 0));
    }
    norm2[0] *= delta_y * grid->length_x / (grid->global_no_halo_dim_x - (grid->coordinate_system == "cylindrical" ? 1 : 0));
    energy_expected_values_updated = true;
    delete [] norm2_kin;
}

double Solver::get_total_energy(void) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    return total_energy;
}

double Solver::get_squared_norm(size_t which) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (which == 3)
        if (single_component)
            return norm2[0];
        else
            return norm2[0] + norm2[1];
    else if (which == 1)
        return norm2[0];
    else if (which == 2)
        if (single_component) {
            cout << "The system has only one component. No input have to be given\n";
            return 0;
        }
        else {
            return norm2[1];
        }
    else {
        cout << "Input may be 1, 2 or 3\n";
        return 0;
    }
}

double Solver::get_kinetic_energy(size_t which) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (which == 3)
        return tot_kinetic_energy;
    else if (which == 1)
        return kinetic_energy[0];
    else if (which == 2)
        if (single_component) {
            cout << "The system has only one component. No input have to be given\n";
            return 0;
        }
        else
            return kinetic_energy[1];
    else {
        cout << "Input may be 1, 2 or 3\n";
        return 0;
    }
}

double Solver::get_potential_energy(size_t which) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (which == 3)
        return tot_potential_energy;
    else if (which == 1)
        return potential_energy[0];
    else if (which == 2)
        if (single_component) {
            cout << "The system has only one component. No input have to be given\n";
            return 0;
        }
        else
            return potential_energy[1];
    else {
        cout << "Input may be 1, 2 or 3\n";
        return 0;
    }
}

double Solver::get_rotational_energy(size_t which) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (which == 3)
        return tot_rotational_energy;
    else if (which == 1)
        return rotational_energy[0];
    else if (which == 2)
        if (single_component) {
            cout << "The system has only one component. No input have to be given\n";
            return 0;
        }
        else
            return rotational_energy[1];
    else {
        cout << "Input may be 1, 2 or 3\n";
        return 0;
    }
}

double Solver::get_intra_species_energy(size_t which) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (which == 3)
        return tot_intra_species_energy;
    else if (which == 1)
        return intra_species_energy[0];
    else if (which == 2)
        if (single_component) {
            cout << "The system has only one component. No input have to be given\n";
            return 0;
        }
        else
            return intra_species_energy[1];
    else {
        cout << "Input may be 1, 2 or 3\n";
        return 0;
    }
}

double Solver::get_LeeHuangYang_energy(void) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    return LeeHuangYang_energy;
}

double Solver::get_inter_species_energy(void) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (!single_component)
        return inter_species_energy;
    else {
        cout << "The system has only one component\n";
        return 0;
    }
}

double Solver::get_rabi_energy(void) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (!single_component)
        return rabi_energy;
    else {
        cout << "The system has only one component\n";
        return 0;
    }
}

void Solver::update_parameters() {
    has_parameters_changed = true;
}
