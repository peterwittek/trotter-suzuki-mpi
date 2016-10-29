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
#include <sstream>
#include <stdexcept>
#include "trottersuzuki.h"
#include "trottersuzuki1D.h"

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

void print_matrix(const char * filename, double * matrix, size_t stride, size_t width, size_t height) {
    ofstream out(filename, ios::out | ios::trunc);
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
    /*
    // output real matrix
    //conversion
    data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y) * charspernum];
    count = 0;
    for (int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
            sprintf(&data_as_txt[count * charspernum], "%+.5e  ", state->p_real[j * grid->dim_x + k]);
            count++;
        }
        if(grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
            sprintf(&data_as_txt[count * charspernum], "%+.5e\n ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
        else {
            sprintf(&data_as_txt[count * charspernum], "%+.5e  ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
    }

    // open the file, and set the view
    output_filename.str("");
    output_filename << fileprefix << "-iter-real.dat";
    MPI_File_open(grid->cartcomm, const_cast<char*>(output_filename.str().c_str()),
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, localarray, (char *)"native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x) * ( grid->inner_end_y - grid->inner_start_y), num_as_string, &status);
    MPI_File_close(&file);
    delete [] data_as_txt;
    */
#else
    stringstream output_filename;
    /*
    output_filename << fileprefix << "-iter-real.dat";
    print_matrix(output_filename.str().c_str(), &(state->p_real[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
                 grid->global_dim_x - 2 * grid->periods[1]*grid->halo_x, grid->global_dim_y - 2 * grid->periods[0]*grid->halo_y);
    */
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

void stamp1D(Lattice1D *grid, State1D *state, string fileprefix) {
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
	int globalsizes[1] = { grid->global_dim_x - 2 * grid->periods[0] * grid->halo_x };
	int localsizes[1] = { grid->inner_end_x - grid->inner_start_x };
	int starts[1] = { grid->inner_start_x };
	int order = MPI_ORDER_C;

	MPI_Datatype complex_localarray;
	MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, complex_num_as_string, &complex_localarray);
	MPI_Type_commit(&complex_localarray);
	MPI_Datatype localarray;
	MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
	MPI_Type_commit(&localarray);

	// output complex matrix
	// conversion
	data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) * chars_per_complex_num];
	count = 0;

	for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
		sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)   ", state->p_real[ k], state->p_imag[ k]);
		count++;
	}
	if (grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
		sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)\n  ", state->p_real[ (grid->inner_end_x - grid->start_x) - 1], state->p_imag[ (grid->inner_end_x - grid->start_x) - 1]);
		count++;
	}
	else {
		sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)   ", state->p_real[ (grid->inner_end_x - grid->start_x) - 1], state->p_imag[ (grid->inner_end_x - grid->start_x) - 1]);
		count++;
	}

	// open the file, and set the view
	stringstream output_filename;
	output_filename << fileprefix;
	MPI_File_open(grid->cartcomm, const_cast<char*>(output_filename.str().c_str()),
		MPI_MODE_CREATE | MPI_MODE_WRONLY,
		MPI_INFO_NULL, &file);

	MPI_File_set_view(file, 0, MPI_CHAR, complex_localarray, (char *)"native", MPI_INFO_NULL);

	MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x), complex_num_as_string, &status);
	MPI_File_close(&file);
	delete[] data_as_txt;
	/*
	// output real matrix
	//conversion
	data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y) * charspernum];
	count = 0;
	for (int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
	for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
	sprintf(&data_as_txt[count * charspernum], "%+.5e  ", state->p_real[j * grid->dim_x + k]);
	count++;
	}
	if(grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
	sprintf(&data_as_txt[count * charspernum], "%+.5e\n ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
	count++;
	}
	else {
	sprintf(&data_as_txt[count * charspernum], "%+.5e  ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
	count++;
	}
	}

	// open the file, and set the view
	output_filename.str("");
	output_filename << fileprefix << "-iter-real.dat";
	MPI_File_open(grid->cartcomm, const_cast<char*>(output_filename.str().c_str()),
	MPI_MODE_CREATE | MPI_MODE_WRONLY,
	MPI_INFO_NULL, &file);

	MPI_File_set_view(file, 0,  MPI_CHAR, localarray, (char *)"native", MPI_INFO_NULL);

	MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x) * ( grid->inner_end_y - grid->inner_start_y), num_as_string, &status);
	MPI_File_close(&file);
	delete [] data_as_txt;
	*/
#else
	stringstream output_filename;
	/*
	output_filename << fileprefix << "-iter-real.dat";
	print_matrix(output_filename.str().c_str(), &(state->p_real[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
	grid->global_dim_x - 2 * grid->periods[1]*grid->halo_x, grid->global_dim_y - 2 * grid->periods[0]*grid->halo_y);
	*/
	output_filename.str("");
	output_filename << fileprefix;
	print_complex_matrix(output_filename.str().c_str(), &(state->p_real[grid->global_dim_x + grid->inner_start_x - grid->start_x]), &(state->p_imag[grid->global_dim_x + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
		grid->global_dim_x - 2 * grid->periods[0] * grid->halo_x,1);
#endif
	return;
}

void stamp_matrix1D(Lattice1D *grid, double *matrix, string filename) {

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
	int globalsizes[1] = { grid->global_dim_x - 2 * grid->periods[0] * grid->halo_x };
	int localsizes[1] = { grid->inner_end_x - grid->inner_start_x };
	int starts[1] = { grid->inner_start_x };
	int order = MPI_ORDER_C;

	MPI_Datatype localarray;
	MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
	MPI_Type_commit(&localarray);

	// output real matrix
	//conversion
	data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) *  charspernum];
	count = 0;

	for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
		sprintf(&data_as_txt[count * charspernum], "%+.5e  ", matrix[ k]);
		count++;
	}
	if (grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
		sprintf(&data_as_txt[count * charspernum], "%+.5e\n ", matrix[ (grid->inner_end_x - grid->start_x) - 1]);
		count++;
	}
	else {
		sprintf(&data_as_txt[count * charspernum], "%+.5e  ", matrix[ (grid->inner_end_x - grid->start_x) - 1]);
		count++;
	}


	// open the file, and set the view
	MPI_File_open(grid->cartcomm, const_cast<char*>(filename.c_str()),
		MPI_MODE_CREATE | MPI_MODE_WRONLY,
		MPI_INFO_NULL, &file);

	MPI_File_set_view(file, 0, MPI_CHAR, localarray, (char *)"native", MPI_INFO_NULL);

	MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x), num_as_string, &status);
	MPI_File_close(&file);
	delete[] data_as_txt;
#else
	print_matrix(filename.c_str(), &(matrix[grid->global_dim_x + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
		grid->global_dim_x - 2 * grid->periods[0] * grid->halo_x,1);
#endif
	return;
}
