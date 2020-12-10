from cosmolisa.galaxy cimport Galaxy
from cosmolisa.cosmology cimport CosmologicalParameters

cdef double log_add(double x, double y)

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
                                    double sigma_z = *)
                                    
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
                                    double sigma_z)
                                    
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
                                    double sigma_z = *)
                                    
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
                                    double sigma_z)

cdef double ComputeLogLhNoEmission(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    object Schechter,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double M_max,
                                    double M_min,
                                    double m_th)
                                    
cdef double _ComputeLogLhNoEmission(Galaxy gal,
                                    object event,
                                    CosmologicalParameters omega,
                                    object Schechter,
                                    double zmin,
                                    double zmax,
                                    double M_cutoff,
                                    double M_max,
                                    double M_min,
                                    double m_th)

cdef unsigned int ComputeEmitters(unsigned int N_tot, object Schechter, double M_cutoff, double M_min, double M_max, int n = *)

cdef unsigned int ComputeBright(double numberdensity, CosmologicalParameters omega, object Schechter, double m_th, double zmin, double zmax, double M_min, double M_max, int n = *)


cdef double prob_Nobs(unsigned int N_obs, unsigned int N_b, str distribution = *)

cdef double _LumDist(double z, CosmologicalParameters omega)

cdef double _dLumDist(double z, CosmologicalParameters omega)

cdef double ComputeRedshift(double LD, CosmologicalParameters omega, double zinit = *, double limit = *)

cdef double ComputeRedshiftFromCoVol(double CoVol, CosmologicalParameters omega, double zinit = *, double limit = *)
