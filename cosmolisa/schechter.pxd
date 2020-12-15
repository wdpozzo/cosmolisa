cdef class SchechterMagFunctionInternal:
    cdef public double Mstar
    cdef public double phistar
    cdef public double alpha
    cdef public double mmin
    cdef public double mmax
    cdef public double norm
    cpdef double _evaluate(self, double m)
    cdef double normalise(self)
    cpdef double pdf(self, double m)

cpdef tuple SchechterMagFunction(double mmin, double mmax, double h = *, str band = *, double phistar = *)

cdef double M_Mobs(double h, double M_obs)
