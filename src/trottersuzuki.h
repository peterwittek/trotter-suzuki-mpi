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
#ifndef __TROTTERSUZUKI_H
#define __TROTTERSUZUKI_H

#include <string>
#define _USE_MATH_DEFINES
#include <cfloat>
#include <complex>
#if HAVE_CONFIG_H
#include "config.h"
#endif
#ifdef HAVE_MPI
#include <mpi.h>
#endif

using namespace std;

class Lattice {
public:
    int mpi_rank;    ///< Rank of the process in the MPI topology.
    int mpi_procs;    ///< Number of processes in the MPI topology.
    double length_x, length_y;    ///< Physical length of the lattice's sides.
    double delta_x, delta_y;    ///< Physical distance between two consecutive point of the grid, along the x and y axes.
    int dim_x, dim_y;    ///< Linear dimension of the tile along x and y axes.
    int global_no_halo_dim_x, global_no_halo_dim_y;    ///< Linear dimension of the lattice, excluding the eventual surrounding halo, along x and y axes.
    int global_dim_x, global_dim_y;    ///< Linear dimension of the lattice, comprising the eventual surrounding halo, along x and y axes.
    int periods[2];    ///< Whether the grid is periodic in any of the directions.
    string coordinate_system;	///< Type of the coordinate system used.

    // Computational topology
    int halo_x, halo_y;    ///< Halo length along the x and y halos.
    int start_x, start_y;    ///< Spatial coordinates (not physical) of the first element of the tile.
    int end_x, end_y;    ///< Spatial coordinates (not physical) of the last element of the tile.
    int inner_start_x, inner_start_y;    ///< Spatial coordinates (not physical) of the first element of the tile, excluding the eventual surrounding halo.
    int inner_end_x, inner_end_y;    ///< Spatial coordinates (not physical) of the last element of the tile, excluding the eventual surrounding halo.
    int mpi_coords[2], mpi_dims[2];    ///< Coordinate of the process in the MPI topology and structure of the MPI topology.
#ifdef HAVE_MPI
    MPI_Comm cartcomm;    ///< MPI communitaros chart.
#endif
};

/**
 * \brief This class defines the 1D lattice structure over which the state and potential are defined.
 *
 * As to single-process execution, the lattice is a single tile which can be surrounded by a halo, in the case of periodic boundary conditions.
 * As to multi-process execution, the lattice is divided in smaller lattices, dubbed tiles, one for each process. Each of the tiles is surrounded by a halo.
 */
class Lattice1D: public Lattice {
public:
    /**
        Lattice constructor.

        @param [in] dim               Linear dimension of the chain lattice.
        @param [in] length            Physical length of the lattice.
        @param [in] periodic_x_axis   Boundary condition along the x axis (false=closed, true=periodic).
        @param [in] coordinate_system Type of the coordinate system used.
     */
    Lattice1D(int dim, double length, bool periodic_x_axis = false, string coordinate_system = "cartesian");
};

/**
 * \brief This class defines the 2D lattice structure over which the state and potential matrices are defined.
 *
 * As to single-process execution, the lattice is a single tile which can be surrounded by a halo, in the case of periodic boundary conditions.
 * As to multi-process execution, the lattice is divided in smaller lattices, dubbed tiles, one for each process. Each of the tiles is surrounded by a halo.
 */
class Lattice2D: public Lattice {
public:
    /**
        Lattice constructor.

        @param [in] dim               Linear dimension of the squared lattice.
        @param [in] length            Physical length of the lattice's side
        @param [in] periodic_x_axis   Boundary condition along the x axis (false=closed, true=periodic).
        @param [in] periodic_y_axis   Boundary condition along the y axis (false=closed, true=periodic).
        @param [in] angular_velocity  Angular velocity of the frame of reference.
        @param [in] coordinate_system Type of the coordinate system used.
     */
    Lattice2D(int dim, double length,
              bool periodic_x_axis = false, bool periodic_y_axis = false,
              double angular_velocity = 0., string coordinate_system = "cartesian");
    /**
        Lattice constructor.

        @param [in] dim_x             Linear dimension in x direction
        @param [in] length_x          Physical length of the lattice's side along the x axis.
        @param [in] dim_y             Linear dimension in y direction
        @param [in] length_y          Physical length of the lattice's side along the y axis.
        @param [in] periodic_x_axis   Boundary condition along the x axis (false=closed, true=periodic).
        @param [in] periodic_y_axis   Boundary condition along the y axis (false=closed, true=periodic).
        @param [in] angular_velocity  Angular velocity of the frame of reference.
        @param [in] coordinate_system Type of the coordinate system used.
     */
    Lattice2D(int dim_x, double length_x, int dim_y, double length_y,
              bool periodic_x_axis = false, bool periodic_y_axis = false,
              double angular_velocity = 0., string coordinate_system = "cartesian");
private:
    void init(int dim_x, double length_x, int dim_y, double length_y,
              bool periodic_x_axis = false, bool periodic_y_axis = false,
              double angular_velocity = 0., string coordinate_system = "cartesian");
};

/**
 * \brief This class defines the quantum state.
 */

class State {
public:
    double *p_real;    ///< Real part of the wave function.
    double *p_imag;    ///< Imaginary part of the wave function.
    Lattice *grid;    ///< Object that defines the lattice structure.
    int angular_momentum;   ///< Angular momentum when cylindrical coordinates are used.

    /**
        Construct the state from given matrices if they are provided, otherwise construct a state with null wave function, initializing p_real and p_imag.

        @param [in] grid             Lattice object.
        @param [in] angular_momentum Angular momentum (only for cylindrical coordinates).
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    State(Lattice *grid, int angular_momentum = 0, double *p_real = 0, double *p_imag = 0);
    State(const State &obj /**< [in] State object. */);    ///< Copy constructor: copy the state object.
    ~State();    ///< Destructor.
    void init_state(complex<double> (*ini_state)(double x) /** Pointer to a wave function */); ///< Write the wave function from a C++ function to p_real and p_imag matrices in 1D.
    void init_state(complex<double> (*ini_state)(double x, double y) /** Pointer to a wave function */);    ///< Write the wave function from a C++ function to p_real and p_imag matrices in 2D.
    void loadtxt(const char *file_name);    ///< Load the wave function from a file to p_real and p_imag matrices.

    void imprint(complex<double> (*function)(double x) /** Pointer to a function */);    ///< Multiply the wave function of the state by the function provided in 1D.
    void imprint(complex<double> (*function)(double x, double y) /** Pointer to a function */);    ///< Multiply the wave function of the state by the function provided in 2D.
    double *get_particle_density(double *density = 0 /** [out] matrix storing the squared norm of the wave function. */);  ///< Return a matrix storing the squared norm of the wave function.
    double *get_phase(double *phase = 0 /** [out] matrix storing the phase of the wave function. */);  ///< Return a matrix storing the phase of the wave function.
    double get_expected_value(string _operator);    ///< Return the expected value of the operator, given as argument.
    double get_squared_norm(void);    ///< Return the squared norm of the quantum state.
    double get_mean_x(void);    ///< Return the expected value of the X operator.
    double get_mean_xx(void);    ///< Return the expected value of the X^2 operator.
    double get_mean_y(void);    ///< Return the expected value of the Y operator.
    double get_mean_yy(void);    ///< Return the expected value of the Y^2 operator.
    double get_mean_px(void);    ///< Return the expected value of the P_x operator.
    double get_mean_pxpx(void);    ///< Return the expected value of the P_x^2 operator.
    double get_mean_py(void);    ///< Return the expected value of the P_y operator.
    double get_mean_pypy(void);    ///< Return the expected value of the P_y^2 operator.
    double get_mean_angular_momentum(void);    ///< Return the expected value of the L_z operator.

    void write_to_file(string fileprefix /** [in] prefix name of the file */);    ///< Write to a file the wave function.
    void write_particle_density(string fileprefix /** [in] prefix name of the file */);    ///< Write to a file the squared norm of the wave function.
    void write_phase(string fileprefix /** [in] prefix name of the file */);    ///< Write to a file the phase of the wave function.
    bool expected_values_updated;    ///< Whether the expected values of the state object are updated with respect to the last evolution.

protected:
    bool self_init;    ///< Whether the p_real and p_imag matrices have been initialized from the State constructor or not.
    void calculate_expected_values(void);    ///< Calculate squared norm and expected values.
    double mean_X, mean_XX;    ///< Expected values of the X and X^2 operators.
    double mean_Y, mean_YY;    ///< Expected values of the Y and Y^2 operators.
    double mean_Px, mean_PxPx;    ///< Expected values of the P_x and P_x^2 operators.
    double mean_Py, mean_PyPy;    ///< Expected values of the P_y and P_y^2 operators.
    double mean_angular_momentum;    ///< Expected value of the L_z operator.
    double norm2;    ///< Squared norm of the state.
};

/**
 * \brief This class defines a quantum state with Bessel function of the first kind, along the radial coordinate.
 *
 * This class is a child of State class. The wave function, along the y-axis, is sinusoidal.
 */
class BesselState: public State {
public:
    /**
    	Construct the quantum state with bessel function of the first kind, along the radial coordinate.

    	@param [in] grid             Lattice object.
    	@param [in] angular_momentum Angular momentum.
    	@param [in] zeros            Number of zeros points along the radial axis.
    	@param [in] norm             Squared norm of the quantum state.
    	@param [in] phase            Relative phase of the wave function.
    	@param [in] p_real           Pointer to the real part of the wave function.
    	@param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    BesselState(Lattice1D *grid, int angular_momentum = 0, int zeros = 1, double norm = 1, double phase = 0, double *p_real = 0, double *p_imag = 0);

    /**
        Construct the quantum state with bessel function of the first kind, along the radial coordinate.

        @param [in] grid             Lattice object.
        @param [in] angular_momentum Angular momentum.
        @param [in] zeros            Number of zeros points along the radial axis.
        @param [in] n_y              Second quantum number for the sinusoid along the y-axis.
        @param [in] norm             Squared norm of the quantum state.
        @param [in] phase            Relative phase of the wave function.
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    BesselState(Lattice2D *grid, int angular_momentum = 0, int zeros = 1, int n_y = 0, double norm = 1, double phase = 0, double *p_real = 0, double *p_imag = 0);

private:
    int angular_momentum;   ///< Angular momentum.
    int zeros, n_y;    ///< First and second quantum number.
    double zero;    ///< radial coordinate of the last zero.
    double normalization;    ///< Variable used for the normalization of the state.
    double norm, phase;    ///< Norm and phase of the state.
    complex<double> bessel_state1D(double x);    ///< Bessel function of the first kind.
    complex<double> bessel_state2D(double x, double y);    ///< Bessel function of the first kind.
};

/**
 * \brief This class defines a quantum state with exponential like wave function.
 *
 * This class is a child of State class.
 */
class ExponentialState: public State {
public:
    /**
    	Construct the quantum state with exponential like wave function.

    	@param [in] grid             Lattice object.
    	@param [in] n_x              First quantum number.
    	@param [in] norm             Squared norm of the quantum state.
    	@param [in] phase            Relative phase of the wave function.
    	@param [in] p_real           Pointer to the real part of the wave function.
    	@param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    ExponentialState(Lattice1D *grid, int n_x = 1, double norm = 1, double phase = 0, double *p_real = 0, double *p_imag = 0);

    /**
        Construct the quantum state with exponential like wave function.

        @param [in] grid             Lattice object.
        @param [in] n_x              First quantum number.
        @param [in] n_y              Second quantum number.
        @param [in] norm             Squared norm of the quantum state.
        @param [in] phase            Relative phase of the wave function.
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    ExponentialState(Lattice2D *grid, int n_x = 1, int n_y = 1, double norm = 1, double phase = 0, double *p_real = 0, double *p_imag = 0);

private:
    int n_x, n_y;    ///< First and second quantum number.
    double norm, phase;    ///< Norm and phase of the state.
    complex<double> exp_state(double x, double y);    ///< Exponential wave function.
};

/**
 * \brief This class defines a quantum state with gaussian like wave function.
 *
 * This class is a child of State class.
 */
class GaussianState: public State {
public:
    /**
    	Construct the quantum state with gaussian like wave function.

    	@param [in] grid             Lattice object.
    	@param [in] omega_x          Inverse of the variance along x-axis.
    	@param [in] mean_x           X coordinate of the gaussian function's center.
    	@param [in] norm             Squared norm of the state.
    	@param [in] phase            Relative phase of the wave function.
    	@param [in] p_real           Pointer to the real part of the wave function.
    	@param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    GaussianState(Lattice1D *grid, double omega_x, double mean_x = 0, double norm = 1, double phase = 0,
                  double *p_real = 0, double *p_imag = 0);

    /**
        Construct the quantum state with gaussian like wave function.

        @param [in] grid             Lattice object.
        @param [in] omega_x          Inverse of the variance along x-axis.
        @param [in] omega_y          Inverse of the variance along y-axis.
        @param [in] mean_x           X coordinate of the gaussian function's center.
        @param [in] mean_y           Y coordinate of the gaussian function's center.
        @param [in] norm             Squared norm of the state.
        @param [in] phase            Relative phase of the wave function.
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    GaussianState(Lattice2D *grid, double omega_x, double omega_y = -1., double mean_x = 0, double mean_y = 0, double norm = 1, double phase = 0,
                  double *p_real = 0, double *p_imag = 0);

private:
    double mean_x;    ///< X coordinate of the gaussian function's center.
    double mean_y;    ///< Y coordinate of the gaussian function's center.
    double omega_x;    ///< Gaussian coefficient.
    double omega_y;    ///< Gaussian coefficient.
    double norm;    ///< Norm of the state.
    double phase;    ///< Relative phase of the wave function.
    complex<double> gauss_state(double x, double y);    ///< Gaussian function.
};

/**
 * \brief This class defines a quantum state with sinusoidal like wave function.
 *
 * This class is a child of State class.
 */
class SinusoidState: public State {
public:
    /**
    	Construct the quantum state with sinusoidal like wave function.

    	@param [in] grid             Lattice object.
    	@param [in] n_x              First quantum number.
    	@param [in] norm             Squared norm of the quantum state.
    	@param [in] phase            Relative phase of the wave function.
    	@param [in] p_real           Pointer to the real part of the wave function.
    	@param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    SinusoidState(Lattice1D *grid, int n_x = 1, double norm = 1, double phase = 0, double *p_real = 0, double *p_imag = 0);

    /**
        Construct the quantum state with sinusoidal like wave function.

        @param [in] grid             Lattice object.
        @param [in] n_x              First quantum number.
        @param [in] n_y              Second quantum number.
        @param [in] norm             Squared norm of the quantum state.
        @param [in] phase            Relative phase of the wave function.
        @param [in] p_real           Pointer to the real part of the wave function.
        @param [in] p_imag           Pointer to the imaginary part of the wave function.
     */
    SinusoidState(Lattice2D *grid, int n_x = 1, int n_y = 1, double norm = 1, double phase = 0, double *p_real = 0, double *p_imag = 0);

private:
    int n_x, n_y;    ///< First and second quantum number.
    double norm, phase;    ///< Norm and phase of the state.
    complex<double> sinusoid_state(double x, double y);    ///< Sinusoidal function.
};

/**
 * \brief This class defines the external potential that is used for Hamiltonian class.
 */
class Potential {
public:
    Lattice *grid;    ///< Object that defines the lattice structure.
    double *matrix;    ///< Matrix storing the potential.

    /**
    	Construct the external potential.

    	@param [in] grid             Lattice object.
    	@param [in] filename         Name of the file that stores the external potential matrix.
     */
    Potential(Lattice *grid, char *filename);
    /**
    	Construct the external potential.

    	@param [in] grid             Lattice object.
    	@param [in] external_pot     Pointer to the external potential matrix.
     */
    Potential(Lattice *grid, double *external_pot = 0);
    /**
    	Construct the external potential.

    	@param [in] grid                   Lattice object.
    	@param [in] potential_function     Pointer to the static external potential function.
     */
    Potential(Lattice *grid, double (*potential_function)(double x, double y));
    /**
    	Construct the 2D time-evolving external potential.

    	@param [in] grid                   Lattice object.
    	@param [in] potential_function     Pointer to the time-dependent external potential function.
     */
    Potential(Lattice *grid, double (*potential_function)(double x, double y, double t), int t = 0);
    virtual ~Potential();
    virtual double get_value(int x); ///< Get the value at the coordinate x in a 1D model.
    virtual double get_value(int x, int y);    ///< Get the value at the coordinate (x,y) in a 2D model.
    bool update(double t);    ///< Update the potential matrix at time t.
    bool updated_potential_matrix;
protected:
    double current_evolution_time;    ///< Amount of time evolved since the beginning of the evolution.
    double (*static_potential)(double x, double y);    ///< Function of the static external potential.
    double (*evolving_potential)(double x, double y, double t);    ///< Function of the time-dependent external potential.
    bool self_init;    ///< Whether the external potential matrix has been initialized from the Potential constructor or not.
    bool is_static;    ///< Whether the external potential is static or time-dependent.
};

/**
 * \brief This class defines the external potential that is used for Hamiltonian class.
 *
 * This class is a child of Potential class.
 */
class HarmonicPotential: public Potential {
public:
    /**
    	Construct the harmonic external potential.

    	@param [in] grid       Lattice object.
    	@param [in] omegax     Frequency along x axis.
    	@param [in] omegay     Frequency along y axis.
    	@param [in] mass       Mass of the particle.
    	@param [in] mean_x     Minimum of the potential along x axis.
    	@param [in] mean_y     Minimum of the potential along y axis.
     */
    HarmonicPotential(Lattice2D *grid, double omegax, double omegay, double mass = 1., double mean_x = 0., double mean_y = 0.);
    ~HarmonicPotential();
    double get_value(int x, int y);    ///< Return the value of the external potential at coordinate (x,y)

private:
    double omegax, omegay;    ///< Frequencies along x and y axis.
    double mass;    ///< Mass of the particle.
    double mean_x, mean_y;    ///< Minimum of the potential along x and y axis.
};

/**
 * \brief This class defines the Hamiltonian of a single component system.
 */
class Hamiltonian {
public:
    Potential *potential;   ///< Potential object.
    double mass;    ///< Mass of the particle.
    double coupling_a;    ///< Coupling constant of intra-particle interaction.
    double LeeHuangYang_coupling_a;    ///< Coupling constant of the Lee-Huang-Yang term.
    double angular_velocity;    ///< The frame of reference rotates with this angular velocity.
    double rot_coord_x;    ///< X coordinate of the center of rotation.
    double rot_coord_y;    ///< Y coordinate of the center of rotation.
    double azimuthal_potential(double x, int angular_momentum);

    /**
    	Construct the Hamiltonian of a single component system.

    	@param [in] grid                Lattice object.
    	@param [in] potential           Potential object.
    	@param [in] mass                Mass of the particle.
    	@param [in] coupling_a          Coupling constant of intra-particle interaction.
    	@param [in] LeeHuangYang_coupling_a          Coupling constant of the Lee-Huang-Yang term of the first component.
    	@param [in] angular_velocity    The frame of reference rotates with this angular velocity.
    	@param [in] rot_coord_x         X coordinate of the center of rotation.
    	@param [in] rot_coord_y         Y coordinate of the center of rotation.
     */
    Hamiltonian(Lattice *grid, Potential *potential = 0, double mass = 1., double coupling_a = 0., double LeeHuangYang_coupling_a = 0.,
                double angular_velocity = 0.,
                double rot_coord_x = 0, double rot_coord_y = 0);
    ~Hamiltonian();

protected:
    bool self_init;    ///< Whether the potential is initialized in the Hamiltonian constructor or not.
    Lattice *grid;    ///< Lattice object.
};

/**
 * \brief This class defines the Hamiltonian of a two component system.
 */
class Hamiltonian2Component: public Hamiltonian {
public:
    double mass_b;    ///< Mass of the second component.
    double coupling_ab;    ///< Coupling constant of the inter-particles interaction.
    double coupling_b;    ///< Coupling constant of the intra-particles interaction of the second component.
    //double LeeHuangYang_coupling_b;    ///< Coupling constant of the Lee-Huang-Yang term of the second component.
    double omega_r;    ///< Real part of the Rabi coupling.
    double omega_i;    ///< Imaginary part of the Rabi coupling.
    Potential *potential_b;    ///< External potential for the second component.
    double azimuthal_potential_b(double x, int angular_momentum);

    /**
    	Construct the Hamiltonian of a two component system.

    	@param [in] grid                Lattice object.
    	@param [in] potential           Potential of the first component.
    	@param [in] potential_b         Potential of the second component.
    	@param [in] mass                Mass of the first-component's particles.
    	@param [in] mass_b              Mass of the second-component's particles.
    	@param [in] coupling_a          Coupling constant of intra-particle interaction for the first component.
    	@param [in] coupling_ab         Coupling constant of inter-particle interaction between the two components.
    	@param [in] coupling_b          Coupling constant of intra-particle interaction of the second component.
    	@param [in] omega_r             Real part of the Rabi coupling.
    	@param [in] omega_i             Imaginary part of the Rabi coupling.
    	@param [in] angular_velocity    The frame of reference rotates with this angular velocity.
    	@param [in] rot_coord_x         X coordinate of the center of rotation.
    	@param [in] rot_coord_y         Y coordinate of the center of rotation.
     */
    Hamiltonian2Component(Lattice *grid, Potential *potential = 0,
                          Potential *potential_b = 0,
                          double mass = 1., double mass_b = 1.,
                          double coupling_a = 0.,/* double LeeHuangYang_coupling_a = 0.,*/ double coupling_ab = 0.,
                          double coupling_b = 0.,/* double LeeHuangYang_coupling_b = 0.,*/
                          double omega_r = 0, double omega_i = 0,
                          double angular_velocity = 0.,
                          double rot_coord_x = 0,
                          double rot_coord_y = 0);
    ~Hamiltonian2Component();
};

/**
 * \brief This class defines the prototipe of the kernel classes: CPU, GPU, Hybrid.
 */
class ITrotterKernel {
public:
    virtual ~ITrotterKernel() {};
    virtual void run_kernel() = 0;    ///< Evolve the remaining blocks in the inner part of the tile.
    virtual void run_kernel_on_halo() = 0;    ///< Evolve blocks of wave function at the edge of the tile. This comprises the halos.
    virtual void wait_for_completion() = 0;    ///< Sincronize all the processes at the end of halos communication. Perform normalization for imaginary time evolution.
    virtual void get_sample(size_t dest_stride, size_t x, size_t y, size_t width, size_t height, double * dest_real, double * dest_imag, double * dest_real2 = 0, double * dest_imag2 = 0) const = 0; ///< Get the evolved wave function.
    virtual void normalization() = 0;    ///< Normalization of the two components wave function.
    virtual void rabi_coupling(double var, double delta_t) = 0;    ///< Perform the evolution regarding the Rabi coupling.
    virtual double calculate_squared_norm(bool global = true) const = 0;  ///< Calculate the squared norm of the wave function.
    virtual bool runs_in_place() const = 0;
    virtual string get_name() const = 0;				///< Get kernel name.
    virtual void update_potential(double *_external_pot_real, double *_external_pot_imag, int which) = 0;    ///< Update the evolution matrix, regarding the external potential, at time t.
    virtual void cpy_first_positive_to_first_negative() = 0;    ///< Copy first points with positive radial coordinates to first points with negative coordinates.

    virtual void start_halo_exchange() = 0;					///< Exchange halos between processes.
    virtual void finish_halo_exchange() = 0;				///< Exchange halos between processes.

};

/**
 * \brief This class defines the evolution tasks.
 */
class Solver {
public:
    Lattice *grid;    ///< Lattice object.
    State *state;    ///< State of the first component.
    State *state_b;    ///< State of the second component.
    Hamiltonian *hamiltonian;    ///< Hamiltonian of the system; either single component or two components.
    double current_evolution_time;    ///< Amount of time evolved since the beginning of the evolution.
    /**
    	Construct the Solver object for a single-component system.

    	@param [in] grid                Lattice object.
    	@param [in] state               State of the system.
    	@param [in] hamiltonian         Hamiltonian of the system.
    	@param [in] delta_t             A single evolution iteration, evolves the state for this time.
    	@param [in] kernel_type         Which kernel to use (either cpu or gpu).
     */
    Solver(Lattice *grid, State *state, Hamiltonian *hamiltonian, double delta_t,
           string kernel_type = "cpu");
    /**
    	Construct the Solver object for a two-component system.

    	@param [in] grid                Lattice object.
    	@param [in] state1              First component's state of the system.
    	@param [in] state2              Second component's state of the system.
    	@param [in] hamiltonian         Hamiltonian of the two-component system.
    	@param [in] delta_t             A single evolution iteration, evolves the state for this time.
    	@param [in] kernel_type         Which kernel to use (either cpu or gpu).
     */
    Solver(Lattice *grid, State *state1, State *state2,
           Hamiltonian2Component *hamiltonian,
           double delta_t, string kernel_type = "cpu");
    ~Solver();
    void evolve(int iterations, bool imag_time = false);  ///< Evolve the state of the system.
    void update_parameters();  ///< Notify the solver if any parameter changed in the Hamiltonian.
    double get_total_energy(void);    ///< Get the total energy of the system.
    double get_squared_norm(size_t which = 3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);  ///< Get the squared norm of the state (default: total wave-function).
    double get_kinetic_energy(size_t which = 3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);  ///< Get the kinetic energy of the system.
    double get_potential_energy(size_t which = 3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);  ///< Get the potential energy of the system.
    double get_rotational_energy(size_t which = 3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);  ///< Get the rotational energy of the system.
    double get_intra_species_energy(size_t which = 3 /** [in] Which = 1(first component); 2 (second component); 3(total state) */);  ///< Get the intra-particles interaction energy of the system.
    double get_LeeHuangYang_energy(void);    ///< Get the LeeHuangYang energy (only first component);
    double get_inter_species_energy(void);    ///< Get the inter-particles interaction energy of the system.
    double get_rabi_energy(void);    ///< Get the Rabi energy of the system.
    void set_exp_potential(double *real, int real_length, double *imag,
                           int imag_length, int which); ///< Set exponential potential directly from Python
private:
    bool imag_time;    ///< Whether the time of evolution is imaginary(true) or real(false).
    double **external_pot_real;    ///< Real part of the evolution operator regarding the external potential.
    double **external_pot_imag;    ///< Imaginary part of the evolution operator regarding the external potential.
    double delta_t;    ///< A single evolution iteration, evolves the state for this time.
    double norm2[2];    ///< Squared norms of the two wave function.
    bool single_component;    ///< Whether the system is single-component(true) or two-components(false).
    string kernel_type;    ///< Which kernel are being used (cpu or gpu).
    ITrotterKernel * kernel;    ///< Pointer to the kernel object.
    void initialize_exp_potential(double time_single_it, int which);    ///< Initialize the evolution operator regarding the external potential.
    void init_kernel();    ///< Initialize the kernel (cpu or gpu).
    double total_energy;    ///< Total energy of the system.
    double kinetic_energy[2];    ///< Kinetic energy for the single components.
    double tot_kinetic_energy;    ///< Total kinetic energy of the system.
    double potential_energy[2];    ///< Potential energy for the single components.
    double tot_potential_energy;    ///< Total potential energy of the system.
    double rotational_energy[2];    ///< Rotational energy for the single components.
    double tot_rotational_energy;    ///< Total Rotational energy of the system.
    double intra_species_energy[2];    ///< Intra-particles interaction energy for the single components.
    double tot_intra_species_energy;    ///< Total intra-particles interaction energy of the system.
    double LeeHuangYang_energy;    ///< LeeHuangYang energy term.
    double inter_species_energy;    ///< Inter-particles interaction energy of the system.
    double rabi_energy;    ///< Rabi energy of the system.
    bool has_parameters_changed;   ///< Keeps track whether the Hamiltonian parameters were changed
    bool energy_expected_values_updated;    ///< Whether the expectation values are updated or not.
    void calculate_energy_expected_values(void);    ///< Calculate all the expectation values and the state's norm.
    bool is_python;
};

double const_potential(double x);    ///< Defines the null potential function in 1D.
double const_potential(double x, double y);    ///< Defines the null potential function in 2D.
void map_lattice_to_coordinate_space(Lattice *grid, int x_in, double *x_out);  ///< Centers the coordinates in 1D.
void map_lattice_to_coordinate_space(Lattice *grid, int x_in, int y_in, double *x_out, double *y_out); ///< Centers the coordinates in 2D.
#endif // __TROTTERSUZUKI_H
