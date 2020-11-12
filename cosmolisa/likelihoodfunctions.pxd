from cosmolisa.galaxy cimport Galaxy
from cosmolisa.cosmology cimport CosmologicalParameters

cdef double log_add(double x, double y) nogil

cdef double ComputeLogLhWithPost(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    object Schechter,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    int N_em,
                                    double m_th,
                                    double M_max,
                                    double M_min,
                                    sigma_z = *)
                                    
cdef double _ComputeLogLhWithPost(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    object Schechter,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    int N_em,
                                    double m_th,
                                    double M_max,
                                    double M_min,
                                    sigma_z)
                                    
cdef double ComputeLogLhNoPost(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    object Schechter,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double m_th,
                                    double M_max,
                                    double M_min,
                                    sigma_z = *)
                                    
cdef double _ComputeLogLhNoPost(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    object Schechter,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double m_th,
                                    double M_max,
                                    double M_min,
                                    sigma_z)

cdef double ComputeLogLhNoEmission(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    object Schechter,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double M_max,
                                    double M_min)
                                    
cdef double _ComputeLogLhNoEmission(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    object Schechter,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double M_max,
                                    double M_min)

cdef unsigned int ComputeEmitters(unsigned int N_tot, object Schechter, double M_cutoff, double M_min)

cdef unsigned int ComputeBright(double n, CosmologicalParameters omega, object Schechter, double m_th, double zmin, double zmax, double M_min)

#cdef double integrand(double M, double z, CosmologicalParameters omega, object Schechter)

#cdef object upperbound(double z, double m_th, CosmologicalParameters omega)

cdef double prob_Nobs(unsigned int N_obs, unsigned int N_b, str distribution = *)
