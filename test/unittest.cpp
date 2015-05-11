/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek, 2010-2012 Carlos Bederián
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

#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <sstream>

#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include "mpi.h"

#include "trotter.h"
#include "common.h"

#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE 0
#define SNAPSHOTS 0

//external potential operator in coordinate representation
void potential_op_coord_representation(float *hamilt_pot, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    float constant = 0.;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            hamilt_pot[i * dimx + j] = constant;
        }
    }
}

void init_state(float *p_real, float *p_imag, int dimx, int dimy, int halo_x, int halo_y, int *periods) {
    double s = 64.0; // FIXME: y esto?
    double L_x = dimx - periods[1] * 2 * halo_x;
    double L_y = dimy - periods[0] * 2 * halo_y;
    double n_x = 1., n_y = 1.;

    for (int y = 1; y <= dimy; y++) {
        for (int x = 1; x <= dimx; x++) {
            //std::complex<float> tmp = std::complex<float>(exp(-(pow(x - 180.0, 2.0) + pow(y - 300.0, 2.0)) / (2.0 * pow(s, 2.0))), 0.0)
            //                      * exp(std::complex<float>(0.0, 0.4 * (x + y - 480.0)));

            std::complex<float> tmp = std::complex<float> (sin(2 * 3.14159 / L_x * (x - periods[1] * halo_x)) * sin(2 * 3.14159 / L_y * (y - periods[0] * halo_y)), 0.0);

            //std::complex<float> tmp = exp(std::complex<float>(0. , 2 * 3.14159 / L_x * (x - periods[1]*halo_x) + 2 * 3.14159 / L_y * (y - periods[0]*halo_y) ));

            p_real[y * dimx + x] = real(tmp);
            p_imag[y * dimx + x] = imag(tmp);
        }
    }
}

void read_initial_state(float *p_real, float *p_imag, int dimx, int dimy, char *file_name, int halo_x, int halo_y, int *periods) {
    std::ifstream input(file_name);

    int in_width = dimx - 2 * periods[1] * halo_x;
    int in_height = dimy - 2 * periods[0] * halo_y;
    std::complex<float> tmp;
    for(int i = 0, idy = periods[0] * halo_y ; i < in_height; i++, idy++) {
        for(int j = 0, idx = periods[1] * halo_x ; j < in_width; j++, idx++) {
            input >> tmp;
            p_real[idy * dimx + idx] = real(tmp);
            p_imag[idy * dimx + idx] = imag(tmp);

            //Down band
            if(i < halo_y && periods[0] != 0) {
                p_real[(idy + in_height) * dimx + idx] = real(tmp);
                p_imag[(idy + in_height) * dimx + idx] = imag(tmp);
                //Down right corner
                if(j < halo_x && periods[1] != 0) {
                    p_real[(idy + in_height) * dimx + idx + in_width] = real(tmp);
                    p_imag[(idy + in_height) * dimx + idx + in_width] = imag(tmp);
                }
                //Down left corner
                if(j >= in_width - halo_x && periods[1] != 0) {
                    p_real[(idy + in_height) * dimx + idx - in_width] = real(tmp);
                    p_imag[(idy + in_height) * dimx + idx - in_width] = imag(tmp);
                }
            }

            //Upper band
            if(i >= in_height - halo_y && periods[0] != 0) {
                p_real[(idy - in_height) * dimx + idx] = real(tmp);
                p_imag[(idy - in_height) * dimx + idx] = imag(tmp);
                //Up right corner
                if(j < halo_x && periods[1] != 0) {
                    p_real[(idy - in_height) * dimx + idx + in_width] = real(tmp);
                    p_imag[(idy - in_height) * dimx + idx + in_width] = imag(tmp);
                }
                //Up left corner
                if(j >= in_width - halo_x && periods[1] != 0) {
                    p_real[(idy - in_height) * dimx + idx - in_width] = real(tmp);
                    p_imag[(idy - in_height) * dimx + idx - in_width] = imag(tmp);
                }
            }
            //Right band
            if(j < halo_x && periods[1] != 0) {
                p_real[idy * dimx + idx + in_width] = real(tmp);
                p_imag[idy * dimx + idx + in_width] = imag(tmp);
            }
            //Left band
            if(j >= in_width - halo_x && periods[1] != 0) {
                p_real[idy * dimx + idx - in_width] = real(tmp);
                p_imag[idy * dimx + idx - in_width] = imag(tmp);
            }
        }
    }
    input.close();
}

//calculate potential part of evolution operator
void init_pot_evolution_op(float * hamilt_pot, float * external_pot_real, float * external_pot_imag, int dimx, int dimy, double particle_mass, double time_single_it ) {
    float CONST_1 = -1. * time_single_it;
    float CONST_2 = 2. * time_single_it / particle_mass;		//CONST_2: discretization of momentum operator and the only effect is to produce a scalar operator, so it could be omitted

    std::complex<float> tmp;
    for(int i = 0; i < dimy; i++) {
        for(int j = 0; j < dimx; j++) {
            tmp = exp(std::complex<float> (0., CONST_1 * hamilt_pot[i * dimx + j] + CONST_2));
            external_pot_real[i * dimx + j] = real(tmp);
            external_pot_imag[i * dimx + j] = imag(tmp);
        }
    }
}

void print_usage() {
    std::cout << "\nTest some functions of CPU and CPU/SSE kernels and simulate\n"\
              "the evolution of a particle in a box.\n\n"\
              "Usage:\n" \
              "     unittest [OPTION]\n" \
              "Arguments:\n" \
              "     -d NUMBER     Matrix dimension (default: " << DIM << ")\n" \
              "     -i NUMBER     Number of iterations (default: " << ITERATIONS << ")\n" \
              "     -k NUMBER     Kernel type (default: " << KERNEL_TYPE << "): \n" \
              "                      0: CPU, cache-optimized\n" \
              "                      1: CPU, SSE and cache-optimized\n" \
              "                      2: GPU\n" \
              "                      3: Hybrid (experimental) \n" \
              "     -s NUMBER     Snapshots are taken at every NUMBER of iterations.\n" \
              "                   Zero means no snapshots. Default: " << SNAPSHOTS << ".\n"\
              "     -n STRING     Set initial state from a file.\n"\
              "     -v            Calculates expected values of energy and momentum operators\n"\
              "                   once the simulation is finished.";
}

void process_command_line(int argc, char** argv, int *dim, int *iterations, int *snapshots, int *kernel_type, bool *values, char *file_name) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;

    int c;
    while ((c = getopt (argc, argv, "d:hi:k:s:vn:")) != -1) {
        switch (c) {
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
            *kernel_type = atoi(optarg);
            if (*kernel_type < 0 || *kernel_type > 3) {
                fprintf (stderr, "The argument of option -k should be a valid kernel.\n");
                abort ();
            }
            break;
        case 's':
            *snapshots = atoi(optarg);
            if (*snapshots <= 0) {
                fprintf (stderr, "The argument of option -s should be a positive integer.\n");
                abort ();
            }
            break;
        case 'v':
            *values = true;
            break;
        case 'n':
            for(int i = 0; i < 100; i++)
                file_name[i] = optarg[i];
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
}

int main(int argc, char** argv) {
    int dim = 0, iterations = 0, snapshots = 0, kernel_type = 0;
    int periods[2] = {1, 1};
    char file_name[100];
    file_name[0] = '\0';
    bool values = false;

    process_command_line(argc, argv, &dim, &iterations, &snapshots, &kernel_type, &values, file_name);

    // Get the top level suite from the registry
    CppUnit::Test *suite = CppUnit::TestFactoryRegistry::getRegistry().makeTest();
    // Adds the test to the list of test to run
    CppUnit::TextUi::TestRunner runner;
    runner.addTest( suite );
    // Change the default outputter to a compiler error format outputter
    runner.setOutputter( new CppUnit::CompilerOutputter( &runner.result(), std::cerr ) );
    // Run the tests.
    bool wasSucessful = runner.run();
    // Return error code 1 if the one of test failed.
    if(!wasSucessful)
        return 1;

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;

    //set hamiltonian variables
    const double particle_mass = 1.;
    float *hamilt_pot = new float[matrix_width * matrix_height];
    potential_op_coord_representation(hamilt_pot, matrix_width, matrix_height, halo_x, halo_y, periods);	//set potential operator

    //set and calculate evolution operator variables from hamiltonian
    const double time_single_it = 0.08 * particle_mass / 2.;	//second approx trotter-suzuki: time/2
    float *external_pot_real = new float[matrix_width * matrix_height];
    float *external_pot_imag = new float[matrix_width * matrix_height];
    init_pot_evolution_op(hamilt_pot, external_pot_real, external_pot_imag, matrix_width, matrix_height, particle_mass, time_single_it);	//calculate potential part of evolution operator
    static const double h_a = cos(time_single_it / (2. * particle_mass));
    static const double h_b = sin(time_single_it / (2. * particle_mass));

    //set initial state
    float *p_real = new float[matrix_width * matrix_height];
    float *p_imag = new float[matrix_width * matrix_height];
    if(file_name[0] == '\0')
        init_state(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, periods);
    else
        read_initial_state(p_real, p_imag, matrix_width, matrix_height, file_name, halo_x, halo_y, periods);

    //set file output directory
    std::stringstream filename;
    std::string filenames;
    if(snapshots) {
        int status = 0;

        filename.str("");
        filename << "D" << dim << "_I" << iterations << "_S" << snapshots << "";
        filenames = filename.str();

        status = mkdir(filenames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if(status != 0 && status != -1)
            filenames = "./";
    }
    else
        filenames = "./";

    std::cout << "Simulation started\n";
    procs_topology var = trotter(h_a, h_b, external_pot_real, external_pot_imag, p_real, p_imag, matrix_width, matrix_height, iterations, snapshots, kernel_type, periods, argc, argv, filenames.c_str());

    if((values == true) && (var.rank == 0)) {
        std::cout << "Calculating expected values\n";
        expect_values(dim, iterations, snapshots, hamilt_pot, particle_mass, filenames.c_str(), var, periods, halo_x, halo_y);
    }

    return 0;
}
