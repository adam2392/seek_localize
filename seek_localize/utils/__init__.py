from .io import (
    read_fieldtrip_elecs,
    MatReader,
)
from .utils import apply_xfm_to_elecs, generate_region_labels, _scale_coordinates
from .freesurfer import _read_vertex_labels, _read_cortex_vertices, convert_fsmesh2mlab
from .space import nearest_electrode_vert
