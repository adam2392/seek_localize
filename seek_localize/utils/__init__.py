from .freesurfer import _read_vertex_labels, _read_cortex_vertices, convert_fsmesh2mlab
from .io import (
    read_fieldtrip_elecs,
    MatReader,
)
from .space import nearest_electrode_vert
from .utils import apply_xfm_to_elecs, generate_region_labels, _scale_coordinates
