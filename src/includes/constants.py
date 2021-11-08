BUCKET = "scl-pipeline"
ZONES = "Zone"  # Attribute used by both zones and gridcells
BIOME = "biome"
COUNTRY = "country"

# data labels coming from sql
DATE = "datestamp"  # consistent label for dates coming from different obs queries
DENSITY = "Density"
GRIDCELL_LOC = "GridCellLocation"
POINT_LOC = "PointLocation"
UNIQUE_ID = "UniqueID"
CT_DAYS_DETECTED = "detections"
CT_DAYS_OPERATING = "Days"
SS_SEGMENTS_DETECTED = "detections"
SS_SEGMENTS_SURVEYED = "NumberOfReplicatesSurveyed"

# labels used in zonify()
EE_ID_LABEL = "id"
MASTER_GRID = "mastergrid"
MASTER_CELL = "mastergridcell"
SCLPOLY_ID = "poly_id"
ZONIFY_DF_COLUMNS = [
    UNIQUE_ID,
    MASTER_GRID,
    MASTER_CELL,
    SCLPOLY_ID,
]

# labels used in classification
HABITAT_AREA = "eff_pot_hab_area"
CONNECTED_HABITAT_AREA = "connected_eff_pot_hab_area"
MIN_PATCHSIZE = "min_patch_size"
PROBABILITY = "phi"
EFFORT = "effort"
RANGE = "range"
