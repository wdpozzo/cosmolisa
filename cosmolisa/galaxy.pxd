cdef class Galaxy:
    """
    Galaxy class:
    initialise a galaxy defined by its redshift, redshift error
    and weight determined by its angular position
    relative to the LISA posterior
    """
    cdef public int ID
    cdef public double RA
    cdef public double DEC
    cdef public double abs_magnitude
    cdef public double app_magnitude
    cdef public double dapp_magnitude
    cdef public double z
    cdef public double dz
    cdef public double weight
    cdef public bint is_detected
    
cpdef object read_galaxy_catalog(dict limits, str catalog_file = *, double n_tot = *)
cdef bint isinbound(Galaxy galaxy, dict limits)
cdef object catalog_weight(object catalog, str weight = *, double ngal = *)
