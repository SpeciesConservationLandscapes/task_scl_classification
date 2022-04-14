BUCKET = "scl-pipeline"
ZONES = "Zone"  # Attribute used by both zones and gridcells
BIOME = "biome"
COUNTRY = "country"
NODATA = -9999

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
DISSOLVED_POLY_ID = "dissolved_poly_id"
PROTECTED = "protected"
ZONIFY_DF_COLUMNS = [
    UNIQUE_ID,
    MASTER_GRID,
    MASTER_CELL,
    SCLPOLY_ID,
    PROTECTED
]

# labels used in classification
HABITAT_AREA = "eff_pot_hab_area"
PROPORTION_PROTECTED = "pa_proportion"
CONNECTED_HABITAT_AREA = "connected_eff_pot_hab_area"
MIN_PATCHSIZE = "min_patch_size"
PROBABILITY = "phi"
PHI0 = "phi0"
EFFORT = "effort"
RANGE = "range"
ADHOC_COUNT_THRESHOLD = 2
