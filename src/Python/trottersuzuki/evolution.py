import numpy as np
from .trottersuzuki import solver, H, K, Lz, Norm2, phase, density


def evolve(p_real, p_imag, particle_mass, external_potential, delta_x, delta_y,
           delta_t, iterations, coupling_const=0.0, kernel_type="cpu",
           periods=None, omega=0.0, rot_coord_x=0.0, rot_coord_y=0.0, 
           imag_time=False):
    """Function for evolving a quantum state.

    :param p_real: The real part of the initial quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the initial quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param particle_mass: Mass of the particle.
    :type particle_mass: float.
    :param external_potential: External potential.
    :type external_potential: 2D numpy.array of float64.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    :param delta_t: Time step.
    :type delta_t: float.
    :param iterations: Number of iterations in the simulation.
    :type iterations: int.
    :param coupling_const: Optional coupling constant between parameters.
    :type coupling_const: float.
    :param kernel_type: Optional parameter to set kernel: cpu, gpu, or hybrid.
    :type kernel_type: str.
    :param periods: Optional parameter to specify periodicity in x and y
                    directions.
    :type periods: [int, int]
    :param omega: angular velocity of the frame system
    :type omega: float.
    :param rot_coord_x: x coordinate of the rotating axis
    :type rot_coord_x: float.
    :param rot_coord_y: y coordinate of the rotating axis
    :type rot_coord_y: float.
    :param imag_time: Optional parameter to request imaginary time evolution.
                      Default: False.
    :type imag_time: bool.
    """
    if external_potential is None:
        external_potential = np.zeros(p_real.shape)
    if periods is None:
        periods = [0, 0]
    solver(p_real, p_imag, particle_mass, coupling_const, external_potential, 
           omega, rot_coord_x, rot_coord_y, delta_x, delta_y, delta_t, 
           iterations, kernel_type, periods, imag_time)


def calculate_total_energy(p_real, p_imag, particle_mass, external_potential, 
                           delta_x, delta_y, coupling_const=0.0, omega=0.0, 
                           rot_coord_x=0.0, rot_coord_y=0.0):
    """Function for calculating the expectation value of the Hamiltonian.

    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param particle_mass: Mass of the particle.
    :type particle_mass: float.
    :param external_potential: External potential.
    :type external_potential: 2D numpy.array of float64.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    :param coupling_const: Optional coupling constant between parameters.
    :type coupling_const: float.
    :param omega: angular velocity of the frame system
    :type omega: float.
    :param rot_coord_x: x coordinate of the rotating axis
    :type rot_coord_x: float.
    :param rot_coord_y: y coordinate of the rotating axis
    :type rot_coord_y: float.
    """
    if external_potential is None:
        external_potential = np.zeros(p_real.shape)
    return H(p_real, p_imag, particle_mass, coupling_const, external_potential, 
             omega, coord_rot_x, coord_rot_y, delta_x, delta_y)


def calculate_kinetic_energy(p_real, p_imag, particle_mass, delta_x, delta_y):
    """Function for calculating the expectation value of the kinetic energy.

    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param particle_mass: Mass of the particle.
    :type particle_mass: float.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    """
    return K(p_real, p_imag, particle_mass, delta_x, delta_y)


def calculate_rotational_energy(p_real, p_imag, delta_x, delta_y, omega=0.0, 
                                rot_coord_x=0.0, rot_coord_y=0.0):
    """Function for calculating rotational energy of a system in a rotating frame of reference. The axis of rotation is parallel to z.
    
    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    :param omega: Angular velocity of the frame system.
    :type omega: float.
    :param coord_rot_x: x-coordinate of the rotation axis.
    :type rot_coord_x: int.
    :param coord_rot_y: y-coordinate of the rotation axis.
    :type rot_coord_x: int.
    """
    return Lz(p_real, p_imag, omega, coord_rot_x, coord_rot_y, delta_x, delta_y)


def calculate_norm2(p_real, p_imag, delta_x, delta_y):
    """Function for calculating the squared norm of the state.

    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    :param delta_x: Relative grid distance in the x direction.
    :type delta_x: float.
    :param delta_y: Relative grid distance in the y direction.
    :type delta_y: float.
    """
    return Norm2(p_real, p_imag, delta_x, delta_y)


def get_wave_function_phase(p_real, p_imag):
    """Function that return the phase of the wave function

    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    """
    phase_matrix = np.zeros(p_real.shape)
    phase(phase_matrix, p_real, p_imag)
    return phase_matrix


def get_wave_function_density(p_real, p_imag):
    """Function that return the particle denity of the wave function

    :param p_real: The real part of the quantum state.
    :type p_real: 2D numpy.array of float64.
    :param p_imag: The imaginary part of the quantum state.
    :type p_imag: 2D numpy.array of float64.
    """
    density_matrix = np.zeros(p_real.shape)
    density(density_matrix, p_real, p_imag)
    return density_matrix
