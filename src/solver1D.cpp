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
#include "trottersuzuki1D.h"
#include "common.h"
#include "kernel1D.h"
#include <iostream>


Solver1D::Solver1D(Lattice1D *_grid, State1D *_state, Hamiltonian1D *_hamiltonian,
               double _delta_t, string _kernel_type):
    grid(_grid), state(_state), hamiltonian(_hamiltonian), delta_t(_delta_t),
    kernel_type(_kernel_type) {
    external_pot_real = new double* [2];
    external_pot_imag = new double* [2];
    external_pot_real[0] = new double[grid->dim_x ];
    external_pot_imag[0] = new double[grid->dim_x ];
    external_pot_real[1] = NULL;
    external_pot_imag[1] = NULL;
    state_b = NULL;
    kernel = NULL;
    current_evolution_time = 0;
    single_component = true;
    energy_expected_values_updated = false;
    has_parameters_changed = false;
}

Solver1D::Solver1D(Lattice1D *_grid, State1D *state1, State1D *state2,
               Hamiltonian2Component1D *_hamiltonian,
               double _delta_t, string _kernel_type):
    grid(_grid), state(state1), state_b(state2), hamiltonian(_hamiltonian), delta_t(_delta_t),
    kernel_type(_kernel_type) {
    external_pot_real = new double* [2];
    external_pot_imag = new double* [2];
    external_pot_real[0] = new double[grid->dim_x  ];
    external_pot_imag[0] = new double[grid->dim_x  ];
    external_pot_real[1] = new double[grid->dim_x  ];
    external_pot_imag[1] = new double[grid->dim_x  ];
    kernel = NULL;
    current_evolution_time = 0;
    single_component = false;
    energy_expected_values_updated = false;
    has_parameters_changed = false;
}

Solver1D::~Solver1D() {
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

void Solver1D::initialize_exp_potential(double delta_t, int which) {
    complex<double> tmp;
    double ptmp;
    double  idx;
        idx = grid->start_x * grid->delta_x;
        for (int x = 0; x < grid->dim_x; x++, idx += grid->delta_x) {
            if (which == 0) {
                ptmp = hamiltonian->potential->get_value(x );
				
            } else {
                ptmp = static_cast<Hamiltonian2Component1D*>(hamiltonian)->potential_b->get_value(x );
            }
            if (imag_time) {
                tmp = exp(complex<double> (-delta_t * ptmp, 0.));
            } else {
                tmp = exp(complex<double> (0., -delta_t * ptmp));
				
            }
            external_pot_real[which][x] = real(tmp);
			
            external_pot_imag[which][x] = imag(tmp);
			
        }
		
}

void Solver1D::init_kernel() {
    if (kernel != NULL) {
        delete kernel;
    }
    if (kernel_type == "cpu") {
        if (single_component) {
            kernel = new CPUBlock1D(grid, state, hamiltonian, external_pot_real[0], external_pot_imag[0], h_a[0], h_b[0], delta_t, norm2[0], imag_time);
        }
        else {
            kernel = new CPUBlock1D(grid, state, state_b, static_cast<Hamiltonian2Component1D*>(hamiltonian), external_pot_real, external_pot_imag, h_a, h_b, delta_t, norm2, imag_time);
        }
    }
    else if (kernel_type == "gpu") {
#ifdef CUDA
        if (hamiltonian->angular_velocity != 0) {
            my_abort("The GPU kernel does not work with nonzero angular velocity.");
        }
        if (single_component) {
            kernel = new CC2Kernel(grid, state, hamiltonian, external_pot_real[0], external_pot_imag[0], h_a[0], h_b[0], delta_t, norm2[0], imag_time);
        }
        else {
            kernel = new CC2Kernel(grid, state, state_b, static_cast<Hamiltonian2Component*>(hamiltonian), external_pot_real, external_pot_imag, h_a, h_b, delta_t, norm2, imag_time);
        }
#else
        my_abort("Compiled without CUDA");
#endif
    }
    else if (!single_component) {
        my_abort("Two-component Hamiltonians only work with the CPU and GPU kernels!");
    }
    else if (kernel_type == "hybrid") {
#ifdef CUDA
        kernel = new HybridKernel(grid, state, hamiltonian, external_pot_real[0], external_pot_imag[0], h_a[0], h_b[0], delta_t, norm2[0], imag_time);
#else
        my_abort("Compiled without CUDA");
#endif
    }
    else {
        my_abort("Unknown kernel");
    }
}

void Solver1D::evolve(int iterations, bool _imag_time) {
    if (_imag_time != imag_time || kernel == NULL || has_parameters_changed) {
        imag_time = _imag_time;
        if (imag_time) {
            h_a[0] = cosh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
            h_b[0] = sinh(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
            initialize_exp_potential(delta_t, 0);
            norm2[0] = state->get_squared_norm();
            if (!single_component) {
                h_a[1] = cosh(delta_t / (4. * static_cast<Hamiltonian2Component1D*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_x));
                h_b[1] = sinh(delta_t / (4. * static_cast<Hamiltonian2Component1D*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_x));
                initialize_exp_potential(delta_t, 1);
                norm2[1] = state_b->get_squared_norm();
            }
        }
        else {
            h_a[0] = cos(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
			
            h_b[0] = sin(delta_t / (4. * hamiltonian->mass * grid->delta_x * grid->delta_x));
			
            initialize_exp_potential(delta_t, 0);
			
            if (!single_component) {
                h_a[1] = cos(delta_t / (4. * static_cast<Hamiltonian2Component1D*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_x));
                h_b[1] = sin(delta_t / (4. * static_cast<Hamiltonian2Component1D*>(hamiltonian)->mass_b * grid->delta_x * grid->delta_x));
            initialize_exp_potential(delta_t, 1);
            }
        }
        init_kernel();
        has_parameters_changed = false;
    }
    // Main loop
    double var = 0.5;
    if (!single_component) {
        kernel->rabi_coupling(var, delta_t);
    }
    var = 1.;
    // Main loop
    for (int i = 0; i < iterations; i++) {
        if (i > 0 && hamiltonian->potential->update(current_evolution_time)) {
            initialize_exp_potential(delta_t, 0);
            kernel->update_potential(external_pot_real[0], external_pot_imag[0]);
        }
        if (!single_component && i > 0) {
            if (static_cast<Hamiltonian2Component1D*>(hamiltonian)->potential_b->update(current_evolution_time)) {
                initialize_exp_potential(delta_t, 1);
                kernel->update_potential(external_pot_real[1], external_pot_imag[1]);
            }
        }
        //first wave function
        kernel->run_kernel_on_halo();
        if (i != iterations - 1) {
            kernel->start_halo_exchange();
        }
        kernel->run_kernel();

        kernel->wait_for_completion();
        if (!single_component) {
            //second wave function
            kernel->run_kernel_on_halo();
            if (i != iterations - 1) {
                kernel->start_halo_exchange();
            }
            kernel->run_kernel();

            kernel->wait_for_completion();
            if (i == iterations - 1) {
                var = 0.5;
            }
            kernel->rabi_coupling(var, delta_t);
            kernel->normalization();
        }
        current_evolution_time += delta_t;
    }
    if (single_component) {
        kernel->get_sample(grid->dim_x, 0, grid->dim_x, state->p_real, state->p_imag);
    }
    else {
        kernel->get_sample(grid->dim_x, 0, grid->dim_x, state->p_real, state->p_imag, state_b->p_real, state_b->p_imag);
        state_b->expected_values_updated = false;
    }
    state->expected_values_updated = false;
    energy_expected_values_updated = false;
}

void Solver1D::calculate_energy_expected_values(void) {

    double sum_norm2_0 = 0;
    double sum_norm2_1 = 0;
    double sum_kinetic_energy_0 = 0;
    double sum_kinetic_energy_1 = 0;
    double sum_potential_energy_0 = 0;
    double sum_potential_energy_1 = 0;
    double sum_intra_species_energy_0 = 0;
    double sum_intra_species_energy_1 = 0;
    double sum_inter_species_energy = 0;
    double sum_rabi_energy = 0;

    int ini_halo_x = grid->inner_start_x - grid->start_x;
    int end_halo_x = grid->end_x - grid->inner_end_x;
    int tile_width = grid->end_x - grid->start_x;


    Potential1D *potential, *potential_b = NULL;
    double coupling, coupling_b = 0, coupling_ab;
    double mass, mass_b;
    complex<double> omega;

    potential = hamiltonian->potential;
    coupling = hamiltonian->coupling_a;
    mass = hamiltonian->mass;
    if (!single_component) {
        potential_b = static_cast<Hamiltonian2Component1D*>(hamiltonian)->potential_b;
        coupling_b = static_cast<Hamiltonian2Component1D*>(hamiltonian)->coupling_b;
        coupling_ab = static_cast<Hamiltonian2Component1D*>(hamiltonian)->coupling_ab;
        mass_b = static_cast<Hamiltonian2Component1D*>(hamiltonian)->mass_b;
        omega = complex<double> (static_cast<Hamiltonian2Component1D*>(hamiltonian)->omega_r,
                                 static_cast<Hamiltonian2Component1D*>(hamiltonian)->omega_i);
    }

    complex<double> cost_E = -1. / (2. * mass);
    complex<double> cost_E_b;
    if (!single_component)
        cost_E_b = complex<double> (-1. / (2. * mass_b));
    complex<double> const_1 = -1. / 12., const_2 = 4. / 3., const_3 = -2.5;
	complex<double> derivate1_1 = 1. / 6., derivate1_2 = -1., derivate1_3 = 0.5, derivate1_4 = 1. / 3.;

#ifndef HAVE_MPI
#pragma omp parallel for reduction(+:sum_norm2_0,\
    sum_potential_energy_0,\
    sum_intra_species_energy_0,\
    sum_kinetic_energy_0,\
    sum_inter_species_energy,\
    sum_rabi_energy,\
    sum_norm2_1,\
    sum_potential_energy_1,\
    sum_intra_species_energy_1,\
    sum_kinetic_energy_1)
#endif

        complex<double>  psi_center, psi_left, psi_right;
        complex<double>  psi_center_b, psi_left_b, psi_right_b;
        complex<double>  psi_left_left, psi_right_right;
        complex<double>  psi_left_left_b, psi_right_right_b;
        int x = grid->inner_start_x;
        for (int j = grid->inner_start_x - grid->start_x; j < grid->inner_end_x - grid->start_x; ++j) {

            psi_center = complex<double> (state->p_real[ j],
                                          state->p_imag[ j]);
            sum_norm2_0 += real(conj(psi_center) * psi_center);
            sum_potential_energy_0 += real(conj(psi_center) * psi_center * complex<double> (potential->get_value(j ), 0.));
            sum_intra_species_energy_0 += real(conj(psi_center) * psi_center * psi_center * conj(psi_center) * complex<double> (0.5 * coupling, 0.));

            if (!single_component) {
                psi_center_b = complex<double> (state_b->p_real[ j],
                                                state_b->p_imag[ j]);

                sum_norm2_1 += real(conj(psi_center_b) * psi_center_b);
                sum_potential_energy_1 += real(conj(psi_center_b) * psi_center_b * complex<double> (potential_b->get_value(j ), 0.));
                sum_intra_species_energy_1 += real(conj(psi_center_b) * psi_center_b * psi_center_b * conj(psi_center_b) * complex<double> (0.5 * coupling_b, 0.));
                sum_inter_species_energy += real(conj(psi_center) * psi_center * conj(psi_center) * psi_center *
                                                 conj(psi_center_b) * psi_center_b * conj(psi_center_b) * psi_center_b * complex<double> (coupling_ab));
                sum_rabi_energy += real(conj(psi_center) * psi_center * conj(psi_center_b) * psi_center_b * (conj(psi_center) * psi_center_b * omega +
                                        conj(psi_center_b * omega) * psi_center));
            }

            if (j - (grid->inner_start_x - grid->start_x) >= (ini_halo_x == 0) * 2 &&
                    j < grid->inner_end_x - grid->start_x - (end_halo_x == 0) * 2) {
                psi_right = complex<double> (state->p_real[ j + 1],
                                             state->p_imag[ j + 1]);
                psi_left = complex<double> (state->p_real[ j - 1],
                                            state->p_imag[ j - 1]);
                psi_right_right = complex<double> (state->p_real[ j + 2],
                                                   state->p_imag[ j + 2]);
                psi_left_left = complex<double> (state->p_real[ j - 2],
                                                 state->p_imag[ j - 2]);

                sum_kinetic_energy_0 += real(conj(psi_center) * cost_E * (complex<double> (1. / (grid->delta_x * grid->delta_x), 0.) *(const_1 * psi_right_right + const_2 * psi_right + const_2 * psi_left + const_1 * psi_left_left + const_3 * psi_center) /*+complex<double> (1. / (grid->delta_y * grid->delta_y), 0.) *(const_1 * psi_down_down + const_2 * psi_down + const_2 * psi_up + const_1 * psi_up_up + const_3 * psi_center)*/));

                if (!single_component) {
                    psi_right_b = complex<double> (state_b->p_real[ j + 1],
                                                   state_b->p_imag[ j + 1]);
                    psi_left_b = complex<double> (state_b->p_real[ j - 1],
                                                  state_b->p_imag[ j - 1]);
                    psi_right_right_b = complex<double> (state_b->p_real[ j + 2],
                                                         state_b->p_imag[ j + 2]);
                    psi_left_left_b = complex<double> (state_b->p_real[ j - 2],
                                                       state_b->p_imag[ j - 2]);

                    sum_kinetic_energy_1 += real(conj(psi_center_b) * cost_E_b * (complex<double> (1. / (grid->delta_x * grid->delta_x), 0.) *(const_1 * psi_right_right_b + const_2 * psi_right_b + const_2 * psi_left_b + const_1 * psi_left_left_b + const_3 * psi_center_b) /*+complex<double> (1. / (grid->delta_y * grid->delta_y), 0.) *(const_1 * psi_down_down_b + const_2 * psi_down_b + const_2 * psi_up_b + const_1 * psi_up_up_b + const_3 * psi_center_b)*/));

                }
            }
            ++x;
        }
    norm2[0] = sum_norm2_0;
    kinetic_energy[0] = sum_kinetic_energy_0;
    potential_energy[0] = sum_potential_energy_0;
    intra_species_energy[0] = sum_intra_species_energy_0;

    if (!single_component) {
    norm2[1] = sum_norm2_1;
        kinetic_energy[1] = sum_kinetic_energy_1;
        potential_energy[1] = sum_potential_energy_1;
        intra_species_energy[1] = sum_intra_species_energy_1;
        inter_species_energy = sum_inter_species_energy;
        rabi_energy = 0.5 * sum_rabi_energy;
    }

#ifdef HAVE_MPI
    double *norm2_mpi = new double[grid->mpi_procs];
    double *kinetic_energy_mpi = new double[grid->mpi_procs];
    double *potential_energy_mpi = new double[grid->mpi_procs];
    double *intra_species_energy_mpi = new double[grid->mpi_procs];

    MPI_Allgather(&norm2[0], 1, MPI_DOUBLE, norm2_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&kinetic_energy[0], 1, MPI_DOUBLE, kinetic_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&potential_energy[0], 1, MPI_DOUBLE, potential_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
    MPI_Allgather(&intra_species_energy[0], 1, MPI_DOUBLE, intra_species_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);

    norm2[0] = 0;
    kinetic_energy[0] = 0;
    potential_energy[0] = 0;
    intra_species_energy[0] = 0;

    for(int i = 0; i < grid->mpi_procs; i++) {
    norm2[0] += norm2_mpi[i];
        kinetic_energy[0] += kinetic_energy_mpi[i];
        potential_energy[0] += potential_energy_mpi[i];
        intra_species_energy[0] += intra_species_energy_mpi[i];
    }
    delete [] norm2_mpi;
    delete [] kinetic_energy_mpi;
    delete [] potential_energy_mpi;
    delete [] intra_species_energy_mpi;

    if (!single_component) {
        double *norm2_mpi = new double[grid->mpi_procs];
        double *kinetic_energy_mpi = new double[grid->mpi_procs];
        double *potential_energy_mpi = new double[grid->mpi_procs];
        double *intra_species_energy_mpi = new double[grid->mpi_procs];
        double *inter_species_energy_mpi = new double[grid->mpi_procs];
        double *rabi_energy_mpi = new double[grid->mpi_procs];

        MPI_Allgather(&norm2[1], 1, MPI_DOUBLE, norm2_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&kinetic_energy[1], 1, MPI_DOUBLE, kinetic_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&potential_energy[1], 1, MPI_DOUBLE, potential_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&intra_species_energy[1], 1, MPI_DOUBLE, intra_species_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&inter_species_energy, 1, MPI_DOUBLE, inter_species_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);
        MPI_Allgather(&rabi_energy, 1, MPI_DOUBLE, rabi_energy_mpi, 1, MPI_DOUBLE, grid->cartcomm);

        norm2[1] = 0;
        kinetic_energy[1] = 0;
        potential_energy[1] = 0;
        intra_species_energy[1] = 0;
        inter_species_energy = 0;
        rabi_energy = 0;

        for(int i = 0; i < grid->mpi_procs; i++) {
            norm2[1] += norm2_mpi[i];
            kinetic_energy[1] += kinetic_energy_mpi[i];
            potential_energy[1] += potential_energy_mpi[i];
            intra_species_energy[1] += intra_species_energy_mpi[i];
            inter_species_energy += inter_species_energy_mpi[i];
            rabi_energy += rabi_energy_mpi[i];
        }
        delete [] norm2_mpi;
        delete [] kinetic_energy_mpi;
        delete [] potential_energy_mpi;
        delete [] intra_species_energy_mpi;
        delete [] inter_species_energy_mpi;
        delete [] rabi_energy_mpi;
    }
#endif
    kinetic_energy[0] = kinetic_energy[0] / norm2[0];
    potential_energy[0] = potential_energy[0] / norm2[0];
    intra_species_energy[0] = intra_species_energy[0] / norm2[0];
    if (single_component) {
    total_energy = kinetic_energy[0] + potential_energy[0] + intra_species_energy[0];
        tot_kinetic_energy = kinetic_energy[0];
        tot_potential_energy = potential_energy[0];
        tot_intra_species_energy = intra_species_energy[0];
    }
    else {
        kinetic_energy[1] = kinetic_energy[1] / norm2[1];
        potential_energy[1] = potential_energy[1] / norm2[1];
        intra_species_energy[1] = intra_species_energy[1] / norm2[1];
        inter_species_energy = inter_species_energy / (norm2[0] * norm2[1]);
        rabi_energy = rabi_energy / (norm2[0] * norm2[1]);

        total_energy = kinetic_energy[0] + potential_energy[0] + intra_species_energy[0] + 
                       kinetic_energy[1] + potential_energy[1] + intra_species_energy[1] + 
                       inter_species_energy + rabi_energy;
        tot_kinetic_energy = kinetic_energy[0] + kinetic_energy[1];
        tot_potential_energy = potential_energy[0] + potential_energy[1];
        tot_intra_species_energy = intra_species_energy[0] + intra_species_energy[1];
        norm2[1] *= grid->delta_x;
    }
    norm2[0] *= grid->delta_x ;
    energy_expected_values_updated = true;
}

double Solver1D::get_total_energy(void) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    return total_energy;
}

double Solver1D::get_squared_norm(size_t which) {
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

double Solver1D::get_kinetic_energy(size_t which) {
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

double Solver1D::get_potential_energy(size_t which) {
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


double Solver1D::get_intra_species_energy(size_t which) {
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

double Solver1D::get_inter_species_energy(void) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (!single_component)
        return inter_species_energy;
    else {
        cout << "The system has only one component\n";
        return 0;
    }
}

double Solver1D::get_rabi_energy(void) {
    if (!energy_expected_values_updated)
        calculate_energy_expected_values();
    if (!single_component)
        return rabi_energy;
    else {
        cout << "The system has only one component\n";
        return 0;
    }
}

void Solver1D::update_parameters() {
    has_parameters_changed = true;
}
