from libc.math cimport INFINITY
cdef class Galaxy:
    """
    Galaxy class:
    initialise a galaxy defined by its redshift, redshift error
    and weight determined by its angular position
    relative to the LISA posterior
    """
    def __init__(self, int ID,
                       double right_ascension,
                       double declination,
                       double redshift,
                       bint is_detected,
                       double rel_z_error = 0.1,
                       double abs_magnitude=INFINITY,
                       double app_magnitude=INFINITY,
                       double weight=0.0):

        self.ID             = ID
        self.RA             = right_ascension
        self.DEC            = declination
        self.abs_magnitude  = abs_magnitude
        self.app_magnitude  = app_magnitude
        self.z              = redshift
        self.dz             = redshift*rel_z_error
        self.weight         = weight
        self.is_detected    = is_detected # Boolean value
