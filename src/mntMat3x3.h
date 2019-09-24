#ifndef MNT_MAT3x3
#define MNT_MAT3x3

#ifndef NO_ASSERT
#include <cassert>
#endif

#include <math.h>
#include <stdlib.h>
#include <vector>
#include "mntVec3.h"
#include "mntVec9.h"

#define MAT3X3_NDIMS 3


/** 3x3 matrix class.
  
  Matrices are column majored, ie
  elements in a column are adjacent in memory.  This
  implementation can be used with Lapack and is compatible with the
  Fortran array layout.
  */

template <class T>
class Matrix3x3 : public Vector9<T> {
  
public:

  /** Constructor with no arguments. */
  Matrix3x3() {};                           
  
  /** Create matrix. Elements are set.
    @param e value of each element
    */
  Matrix3x3(const T& e) {
    for (size_t i = 0; i < this->size(); ++i) {
      (*this)[i] = e;
    }
  }

  /** Indexing operator. 
    @param i the row number
    @param j the column number
    @return reference to an element pointed by i and j
   */
  inline T& operator()(size_t i, size_t j) {
#ifndef NO_ASSERT
    assert(i < MAT3X3_NDIMS);
    assert(j < MAT3X3_NDIMS);
#endif 
    return (*this)[j*MAT3X3_NDIMS + i];
  }

  /** Indexing operator. 
    @param i the row number
    @param j the column number
    @return reference to an element pointed by i and j
   */
  inline const T& operator()(size_t i, size_t j) const {
#ifndef NO_ASSERT
    assert(i < MAT3X3_NDIMS);
    assert(j < MAT3X3_NDIMS);
#endif 
    // column major
    return (*this)[j*MAT3X3_NDIMS + i];
  }

};                                                              

typedef Matrix3x3<double> Mat3x3;


#endif /* MNT_MAT3x3 */

