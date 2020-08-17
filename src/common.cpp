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
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "trottersuzuki.h"
#include "common.h"


void map_lattice_to_coordinate_space(Lattice *grid, int x_in, double *x_out) {
    if (grid->coordinate_system == "cartesian") {
        double idx = grid->start_x * grid->delta_x + 0.5 * grid->delta_x + x_in * grid->delta_x;
        double x_c = grid->global_no_halo_dim_x * grid->delta_x * 0.5;
        if (idx - x_c < -grid->length_x * 0.5) {
            idx += grid->length_x;
        }
        if (idx - x_c > grid->length_x * 0.5) {
            idx -= grid->length_x;
        }
        *x_out = idx - x_c;
    }

    // By convention the radial axis is the x axis.
    if (grid->coordinate_system == "cylindrical") {
        double idx = grid->start_x * grid->delta_x + x_in * grid->delta_x - 0.5 * grid->delta_x;
        *x_out = idx;
    }
}

void map_lattice_to_coordinate_space(Lattice *grid, int x_in, int y_in, double *x_out, double *y_out) {
    if (grid->coordinate_system == "cartesian") {
        double idy = grid->start_y * grid->delta_y + 0.5 * grid->delta_y + y_in * grid->delta_y;
        double idx = grid->start_x * grid->delta_x + 0.5 * grid->delta_x + x_in * grid->delta_x;
        double x_c = grid->global_no_halo_dim_x * grid->delta_x * 0.5;
        double y_c = grid->global_no_halo_dim_y * grid->delta_y * 0.5;
        if (idx - x_c < -grid->length_x * 0.5) {
            idx += grid->length_x;
        }
        if (idx - x_c > grid->length_x * 0.5) {
            idx -= grid->length_x;
        }
        if (idy - y_c < -grid->length_y * 0.5) {
            idy += grid->length_y;
        }
        if (idy - y_c > grid->length_y * 0.5) {
            idy -= grid->length_y;
        }
        *x_out = idx - x_c;
        *y_out = idy - y_c;
    }

    // By convention the radial axis is the x axis.
    if (grid->coordinate_system == "cylindrical") {
        double idy = grid->start_y * grid->delta_y + 0.5 * grid->delta_y + y_in * grid->delta_y;
        double idx = grid->start_x * grid->delta_x + x_in * grid->delta_x - 0.5 * grid->delta_x;
        double y_c = grid->global_no_halo_dim_y * grid->delta_y * 0.5;
        if (idy - y_c < -grid->length_y * 0.5) {
            idy += grid->length_y;
        }
        if (idy - y_c > grid->length_y * 0.5) {
            idy -= grid->length_y;
        }
        *x_out = idx;
        *y_out = idy - y_c;
    }
}

void calculate_borders(int coord, int dim, int * start, int *end, int *inner_start, int *inner_end, int length, int halo, int periodic_bound) {
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

void my_abort(string err) {
#ifdef HAVE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cerr << "Error: " << err << endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    throw std::runtime_error(err);
#endif
}

void add_padding(double *padded_matrix, double *matrix,
                 int padded_dim_x, int padded_dim_y,
                 int halo_x, int halo_y,
                 const int dim_x, const int dim_y, int *periods) {
    for (int i = 0; i < dim_y; i++) {
        for (int j = 0; j < dim_x; j++) {
            double element = matrix[j + i * dim_x];
            padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j + halo_x * periods[1]] = element;
            //Down band
            if (i < halo_y && periods[0] != 0) {
                padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j + halo_x * periods[1]] = element;
                //Down right corner
                if (j < halo_x && periods[1] != 0) {
                    padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j + padded_dim_x - halo_x] = element;
                }
                //Down left corner
                if(j >= dim_x - halo_x && periods[1] != 0) {
                    padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j - (dim_x - halo_x)] = element;
                }
            }
            //Upper band
            if (i >= dim_y - halo_y && periods[0] != 0) {
                padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j + halo_x * periods[1]] = element;
                //Up right corner
                if (j < halo_x && periods[1] != 0) {
                    padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j + padded_dim_x - halo_x] = element;
                }
                //Up left corner
                if (j >= dim_x - halo_x && periods[1] != 0) {
                    padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j - (dim_x - halo_x)] = element;
                }
            }
            //Right band
            if (j < halo_x && periods[1] != 0) {
                padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j + padded_dim_x - halo_x] = element;
            }
            //Left band
            if (j >= dim_x - halo_x && periods[1] != 0) {
                padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j - (dim_x - halo_x)] = element;
            }
        }
    }
}

void print_complex_matrix(const char * filename, double * matrix_real, double * matrix_imag, size_t stride, size_t width, size_t height) {
    ofstream out(filename, ios::out | ios::trunc);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            out << "(" << matrix_real[i * stride + j] << "," << matrix_imag[i * stride + j] << ") ";
        }
        out << endl;
    }
    out.close();
}

void print_matrix(string filename, double * matrix, size_t stride, size_t width, size_t height) {
    ofstream out(filename.c_str(), ios::out | ios::trunc);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            out << matrix[i * stride + j] << " ";
        }
        out << endl;
    }
    out.close();
}

void memcpy2D(void * dst, size_t dstride, const void * src, size_t sstride, size_t width, size_t height) {
    char *d = reinterpret_cast<char *>(dst);
    const char *s = reinterpret_cast<const char *>(src);
//#ifndef HAVE_MPI
//    #pragma omp parallel for schedule(dynamic, 2) collapse(2)
//#endif //wasn't worth it.
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            d[i * dstride + j] = s[i * sstride + j];
        }
    }
}

void stamp(Lattice *grid, State *state, string fileprefix) {
#ifdef HAVE_MPI
    // Set variables for mpi output
    char *data_as_txt;
    int count;

    MPI_File   file;
    MPI_Status status;

    // each number is represented by charspernum chars
    const int chars_per_complex_num = 30;
    MPI_Datatype complex_num_as_string;
    MPI_Type_contiguous(chars_per_complex_num, MPI_CHAR, &complex_num_as_string);
    MPI_Type_commit(&complex_num_as_string);

    const int charspernum = 14;
    MPI_Datatype num_as_string;
    MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string);
    MPI_Type_commit(&num_as_string);

    // create a type describing our piece of the array
    int globalsizes[2] = {grid->global_dim_y - 2 * grid->periods[0] * grid->halo_y, grid->global_dim_x - 2 * grid->periods[1] * grid->halo_x};
    int localsizes [2] = {grid->inner_end_y - grid->inner_start_y, grid->inner_end_x - grid->inner_start_x};
    int starts[2]      = {grid->inner_start_y, grid->inner_start_x};
    int order          = MPI_ORDER_C;

    MPI_Datatype complex_localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, complex_num_as_string, &complex_localarray);
    MPI_Type_commit(&complex_localarray);
    MPI_Datatype localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
    MPI_Type_commit(&localarray);

    // output complex matrix
    // conversion
    data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y) * chars_per_complex_num];
    count = 0;
    for (int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)   ", state->p_real[j * grid->dim_x + k], state->p_imag[j * grid->dim_x + k]);
            count++;
        }
        if(grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)\n  ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1], state->p_imag[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
        else {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)   ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1], state->p_imag[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
    }

    // open the file, and set the view
    stringstream output_filename;
    output_filename << fileprefix;
    MPI_File_open(grid->cartcomm, const_cast<char*>(output_filename.str().c_str()),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, complex_localarray, (char *)"native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y), complex_num_as_string, &status);
    MPI_File_close(&file);
    delete [] data_as_txt;
#else
    stringstream output_filename;
    output_filename.str("");
    output_filename << fileprefix;
    print_complex_matrix(output_filename.str().c_str(), &(state->p_real[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), &(state->p_imag[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
                         grid->global_dim_x - 2 * grid->periods[1]*grid->halo_x, grid->global_dim_y - 2 * grid->periods[0]*grid->halo_y);
#endif
    return;
}

void stamp_matrix(Lattice *grid, double *matrix, string filename) {

#ifdef HAVE_MPI
    // Set variables for mpi output
    char *data_as_txt;
    int count;

    MPI_File   file;
    MPI_Status status;

    // each number is represented by charspernum chars
    const int charspernum = 14;
    MPI_Datatype num_as_string;
    MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string);
    MPI_Type_commit(&num_as_string);

    // create a type describing our piece of the array
    int globalsizes[2] = {grid->global_dim_y - 2 * grid->periods[0] * grid->halo_y, grid->global_dim_x - 2 * grid->periods[1] * grid->halo_x};
    int localsizes [2] = {grid->inner_end_y - grid->inner_start_y, grid->inner_end_x - grid->inner_start_x};
    int starts[2]      = {grid->inner_start_y, grid->inner_start_x};
    int order          = MPI_ORDER_C;

    MPI_Datatype localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
    MPI_Type_commit(&localarray);

    // output real matrix
    //conversion
    data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y) * charspernum];
    count = 0;
    for (int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
            sprintf(&data_as_txt[count * charspernum], "%+.5e  ", matrix[j * grid->dim_x + k]);
            count++;
        }
        if(grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
            sprintf(&data_as_txt[count * charspernum], "%+.5e\n ", matrix[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
        else {
            sprintf(&data_as_txt[count * charspernum], "%+.5e  ", matrix[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
    }

    // open the file, and set the view
    MPI_File_open(grid->cartcomm, const_cast<char*>(filename.c_str()),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, localarray, (char *)"native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y), num_as_string, &status);
    MPI_File_close(&file);
    delete [] data_as_txt;
#else
    print_matrix(filename.c_str(), &(matrix[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
                 grid->global_dim_x - 2 * grid->periods[1]*grid->halo_x, grid->global_dim_y - 2 * grid->periods[0]*grid->halo_y);
#endif
    return;
}

double bessel_j_zeros(int l, int x) {
    // l goes from 0 to 19; x from 0 to 19
    double zeros[] = {2.40482556, 5.52007811, 8.65372791, 11.79153444, 14.93091771, 18.07106397, 21.21163663, 24.35247153, 27.49347913, 30.63460647, 33.77582021, 36.91709835, 40.05842576, 43.19979171, 46.34118837, 49.48260990, 52.62405184, 55.76551076, 58.90698393, 62.04846919,
                      3.83170597, 7.01558667, 10.17346814, 13.32369194, 16.47063005, 19.61585851, 22.76008438, 25.90367209, 29.04682853, 32.18967991, 35.33230755, 38.47476623, 41.61709421, 44.75931900, 47.90146089, 51.04353518, 54.18555364, 57.32752544, 60.46945785, 63.61135670,
                      5.13562230, 8.41724414, 11.61984117, 14.79595178, 17.95981949, 21.11699705, 24.27011231, 27.42057355, 30.56920450, 33.71651951, 36.86285651, 40.00844673, 43.15345378, 46.29799668, 49.44216411, 52.58602351, 55.72962705, 58.87301577, 62.01622236, 65.15927319,
                      6.38016190, 9.76102313, 13.01520072, 16.22346616, 19.40941523, 22.58272959, 25.74816670, 28.90835078, 32.06485241, 35.21867074, 38.37047243, 41.52071967, 44.66974312, 47.81778569, 50.96502991, 54.11161557, 57.25765160, 60.40322414, 63.54840218, 66.69324167,
                      7.58834243, 11.06470949, 14.37253667, 17.61596605, 20.82693296, 24.01901952, 27.19908777, 30.37100767, 33.53713771, 36.69900113, 39.85762730, 43.01373772, 46.16785351, 49.32036069, 52.47155140, 55.62165091, 58.77083574, 61.91924620, 65.06699526, 68.21417486,
                      8.77148382, 12.33860420, 15.70017408, 18.98013388, 22.21779990, 25.43034115, 28.62661831, 31.81171672, 34.98878129, 38.15986856, 41.32638325, 44.48931912, 47.64939981, 50.80716520, 53.96302656, 57.11730278, 60.27024507, 63.42205405, 66.57289189, 69.72289116,
                      9.93610952, 13.58929017, 17.00381967, 20.32078921, 23.58608444, 26.82015198, 30.03372239, 33.23304176, 36.42201967, 39.60323942, 42.77848161, 45.94901600, 49.11577372, 52.27945390, 55.44059207, 58.59960563, 61.75682490, 64.91251478, 68.06689027, 71.22012770,
                      11.08637002, 14.82126873, 18.28758283, 21.64154102, 24.93492789, 28.19118846, 31.42279419, 34.63708935, 37.83871738, 41.03077369, 44.21540851, 47.39416576, 50.56818468, 53.73832537, 56.90524999, 60.06947700, 63.23141837, 66.39140576, 69.54970927, 72.70655117,
                      12.22509226, 16.03777419, 19.55453643, 22.94517313, 26.26681464, 29.54565967, 32.79580004, 36.02561506, 39.24044800, 42.44388774, 45.63844418, 48.82593038, 52.00769146, 55.18474794, 58.35788903, 61.52773517, 64.69478124, 67.85942699, 71.02199904, 74.18276693,
                      13.35430048, 17.24122038, 20.80704779, 24.23388526, 27.58374896, 30.88537897, 34.15437792, 37.40009998, 40.62855372, 43.84380142, 47.04870074, 50.24532696, 53.43522716, 56.61958027, 59.79930163, 62.97511353, 66.14759402, 69.31721152, 72.48434982, 75.64932654,
                      14.47550069, 18.43346367, 22.04698536, 25.50945055, 28.88737506, 32.21185620, 35.49990921, 38.76180702, 42.00419024, 45.23157410, 48.44715139, 51.65325167, 54.85161908, 58.04358793, 61.23019798, 64.41227241, 67.59047207, 70.76533400, 73.93729938, 77.10673425,
                      15.58984788, 19.61596690, 23.27585373, 26.77332255, 30.17906118, 33.52636408, 36.83357134, 40.11182327, 43.36836095, 46.60813268, 49.83465351, 53.05049896, 56.25760472, 59.45745691, 62.65121739, 65.83980880, 69.02397393, 72.20431796, 75.38133933, 78.55545246,
                      16.69824993, 20.78990636, 24.49488504, 28.02670995, 31.45996004, 34.82998699, 38.15637750, 41.45109231, 44.72194354, 47.97429353, 51.21196700, 54.43777693, 57.65384481, 60.86180468, 64.06293782, 67.25826456, 70.44860840, 73.63464196, 76.81692044, 79.99590644,
                      17.80143515, 21.95624407, 25.70510305, 29.27063044, 32.73105331, 36.12365767, 39.46920683, 42.78043927, 46.06571091, 49.33078010, 52.57976906, 55.81571988, 59.04093404, 62.25718939, 65.46588380, 68.66813322, 71.86484051, 75.05674474, 78.24445722, 81.42848829,
                      18.89999795, 23.11577835, 26.90736898, 30.50595016, 33.99318498, 37.40818513, 40.77282785, 44.10059057, 47.40034778, 50.67823695, 53.93866621, 57.18489860, 60.41940985, 63.64411751, 66.86053301, 70.06986583, 73.27309662, 76.47102976, 79.66433188, 82.85356054,
                      19.99443063, 24.26918003, 28.10241523, 31.73341334, 35.24708679, 38.68427639, 42.06791700, 45.41218961, 48.72646412, 52.01724128, 55.28920415, 58.54582890, 61.78975990, 65.02305025, 68.24732200, 71.46387589, 74.67376871, 77.87786897, 81.07689772, 84.27145907,
                      21.08514611, 25.41701901, 29.29087070, 32.95366489, 36.49339791, 39.95255349, 43.35507320, 46.71580944, 50.04460602, 53.34831233, 56.63187594, 59.89897873, 63.15242819, 66.39440904, 69.62665088, 72.85054351, 76.06721817, 79.27760620, 82.48248211, 85.68249584,
                      22.17249462, 26.55978414, 30.47327995, 34.16726785, 37.73268052, 41.21356706, 44.63482975, 48.01196294, 51.35526465, 54.67191918, 57.96712883, 61.24477410, 64.50782040, 67.75858011, 70.99888749, 74.23021912, 77.45377900, 80.67055998, 83.88138905, 87.08696117,
                      23.25677609, 27.69789835, 31.65011815, 35.37471722, 38.96543205, 42.46780721, 45.90766387, 49.30111134, 52.65888365, 55.98848722, 59.29536994, 62.58360418, 65.85630828, 69.11591850, 72.36437087, 75.60322657, 78.83376063, 82.05702611, 85.27390141, 88.48512576,
                      24.33824962, 28.83173035, 32.82180276, 36.57645076, 40.19209510, 43.71571242, 47.17400457, 50.58367114, 53.95586528, 57.29840365, 60.61697113, 63.91582558, 67.19823350, 70.46675142, 73.72341433, 76.96986585, 80.20745037, 83.43727983, 86.66028299, 89.87724253,
                      25.41714081, 29.96160379, 33.98870279, 37.77285784, 41.41306551, 44.95767675, 48.43423920, 51.86001993, 55.24657561, 58.60202207, 61.93227307, 65.24176599, 68.53391094, 71.81138120, 75.07630808, 78.33041549, 81.57511555, 84.81157774, 88.04078020, 91.26354816
                     };
    if (l < 20 && l >= 0 && x >= 0 && x < 20) {
        return zeros[x + l * 20];
    }
    else {
        my_abort("bessel_j_zeros takes integer numbers from 0 to 19");
    }
    return std::numeric_limits<double>::quiet_NaN();
}
