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

#ifndef __COMMON_H
#define __COMMON_H
#include "trottersuzuki.h"
#include "trottersuzuki1D.h"

void stamp(Lattice *grid, State *state, string fileprefix);
void stamp1D(Lattice1D *grid, State1D *state, string fileprefix);

void stamp_matrix(Lattice *grid, double *matrix, string filename);
void stamp_matrix1D(Lattice1D *grid, double *matrix, string filename);

void my_abort(string err);
void memcpy2D(void * dst, size_t dstride, const void * src, size_t sstride, size_t width, size_t height);
#endif
