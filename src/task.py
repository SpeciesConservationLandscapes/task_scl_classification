import argparse
import ee
import os
import re
import subprocess
import numpy as np
import pandas as pd
import pyodbc
import time
import uuid
from typing import Optional, Union

from google.cloud.storage import Client
from google.cloud.exceptions import NotFound
from pathlib import Path
from task_base import SCLTask


class ConversionException(Exception):
    pass


# noinspection PyTypeChecker
class SCLClassification(SCLTask):
    BUCKET = "scl-pipeline"
    DATE_LABEL = "datestamp"
    COUNTRY_LABEL = "country"
    BIOME_LABEL = "biome"
    RANGE_LABEL = "range"
    EXTIRPATED_RANGE_LABEL = "extirpated_range"
    # OBS_DECAY_RATE = -0.001  # for use with age in days
    EE_NODATA = -9999
    ZONES_LABEL = "Biome_zone"
    POINT_LOC_LABEL = "PointLocation"
    GRIDCELL_LOC_LABEL = "GridCellLocation"
    UNIQUE_ID_LABEL = "UniqueID"
    DENSITY_LABEL = "Density"
    CT_DAYS_OPERATING_LABEL = "Days"
    CT_DAYS_DETECTED_LABEL = "detections"
    SS_SEGMENTS_SURVEYED_LABEL = "NumberOfReplicatesSurveyed"
    SS_SEGMENTS_DETECTED_LABEL = "detections"
    MASTER_GRID_LABEL = "mastergrid"
    MASTER_CELL_LABEL = "mastergridcell"
    EE_ID_LABEL = "id"
    EE_SYSTEM_INDEX = "system:index"
    SCLPOLY_ID = "poly_id"
    ZONIFY_DF_COLUMNS = [
        UNIQUE_ID_LABEL,
        MASTER_GRID_LABEL,
        MASTER_CELL_LABEL,
        SCLPOLY_ID,
    ]

    google_creds_path = "/.google_creds"
    inputs = {
        "obs_adhoc": {"maxage": 5},
        "obs_ss": {"maxage": 5},
        "obs_ct": {"maxage": 5},
        "scl": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "scl_polys_path",
            "maxage": 1 / 365,
        },
        "historic_range": {
            "ee_type": SCLTask.IMAGE,
            "ee_path": "projects/SCL/v1/Panthera_tigris/historical_range_img_200914",
            "static": True,
        },
        "extirpated_range": {
            "ee_type": SCLTask.IMAGE,
            "ee_path": "projects/SCL/v1/Panthera_tigris/source/Inputs_2006/extirp_fin",
            "static": True,
        },
        "zones": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/SCL/v1/Panthera_tigris/zones",
            "static": True,
        },
        "gridcells": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "projects/SCL/v1/Panthera_tigris/gridcells",
            "static": True,
        },
        "countries": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "USDOS/LSIB/2017",
            "static": True,
        },
        "ecoregions": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "RESOLVE/ECOREGIONS/2017",
            "static": True,
        },
    }
    thresholds = {
        "landscape_size": 3,
        "current_range": 2,
        "probability": 1,
        "landscape_survey_effort": 1,
    }

    def scenario_habitat(self):
        return f"{self.ee_rootdir}/pothab/potential_habitat"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = kwargs.get("use_cache") or os.environ.get("use_cache") or False

        try:
            self.OBSDB_HOST = os.environ["OBSDB_HOST"]
            self.OBSDB_NAME = os.environ["OBSDB_NAME"]
            self.OBSDB_USER = os.environ["OBSDB_USER"]
            self.OBSDB_PASS = os.environ["OBSDB_PASS"]
        except KeyError as e:
            self.status = self.FAILED
            raise KeyError(str(e)) from e

        self._obsconn_str = (
            f"DRIVER=FreeTDS;SERVER={self.OBSDB_HOST};PORT=1433;DATABASE="
            f"{self.OBSDB_NAME};UID={self.OBSDB_USER};PWD={self.OBSDB_PASS};TDS_VERSION=8.0"
        )

        # Set up google cloud credentials separate from ee creds
        creds_path = Path(self.google_creds_path)
        if creds_path.exists() is False:
            with open(str(creds_path), "w") as f:
                f.write(self.service_account_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_creds_path

        self._df_adhoc = self._df_ct = self._df_ss = None
        self.fc_csvs = []
        self.scl, _ = self.get_most_recent_featurecollection(
            self.inputs["scl"]["ee_path"]
        )
        self.geofilter = ee.Geometry.MultiPolygon(
            self.aoi, proj=self.crs, geodesic=False
        )
        self.zones = ee.FeatureCollection(self.inputs["zones"]["ee_path"])
        self.gridcells = ee.FeatureCollection(
            self.inputs["gridcells"]["ee_path"]
        ).filterBounds(self.geofilter)
        self.countries = ee.FeatureCollection(self.inputs["countries"]["ee_path"])
        self.ecoregions = ee.FeatureCollection(self.inputs["ecoregions"]["ee_path"])
        self.historic_range = ee.Image(self.inputs["historic_range"]["ee_path"]).unmask(
            0
        )
        self.extirpated_range = ee.Image(self.inputs["extirpated_range"]["ee_path"]).eq(
            0
        )
        # 0: neither historic or extirpated range
        # 1: extirpated
        # 2: historic
        self.range_class = (
            ee.Image(0)
            .where(self.historic_range.eq(1), ee.Image(2))
            .where(self.extirpated_range.eq(1), ee.Image(1))
        ).selfMask()
        self.intersects = ee.Filter.intersects(".geo", None, ".geo")

        self.scl_poly_filters = {
            "scl_species": ee.Filter.And(
                ee.Filter.gte("size", self.thresholds["landscape_size"]),
                ee.Filter.eq("range", self.thresholds["current_range"]),
                ee.Filter.gte("probability", self.thresholds["probability"]),
                ee.Filter.gte("effort", self.thresholds["landscape_survey_effort"]),
            ),
            "scl_restoration": ee.Filter.Or(
                ee.Filter.And(
                    ee.Filter.gte("size", self.thresholds["landscape_size"]),
                    ee.Filter.eq("range", self.thresholds["current_range"]),
                    ee.Filter.lt("probability", self.thresholds["probability"]),
                    ee.Filter.gte("effort", self.thresholds["landscape_survey_effort"]),
                ),
                ee.Filter.And(
                    ee.Filter.gte("size", self.thresholds["landscape_size"]),
                    ee.Filter.lt("range", self.thresholds["current_range"]),
                ),
            ),
            "scl_survey": ee.Filter.And(
                ee.Filter.gte("size", self.thresholds["landscape_size"]),
                ee.Filter.eq("range", self.thresholds["current_range"]),
                ee.Filter.gte("probability", self.thresholds["probability"]),
                ee.Filter.lt("effort", self.thresholds["landscape_survey_effort"]),
            ),
            "scl_fragment": ee.Filter.And(
                ee.Filter.lt("size", self.thresholds["landscape_size"]),
                ee.Filter.eq("range", self.thresholds["current_range"]),
                ee.Filter.gte("probability", self.thresholds["probability"]),
                ee.Filter.gte("effort", self.thresholds["landscape_survey_effort"]),
            ),
        }

    def scl_polys_path(self):
        return f"{self.ee_rootdir}/pothab/scl_polys"

    def poly_export(self, polys, scl_name):
        size_test = polys.size().gt(0).getInfo()
        path = "pothab/" + scl_name
        if size_test:
            self.export_fc_ee(polys, path)
        else:
            print("no " + scl_name + " polygons delineated")

    def _download_from_cloudstorage(self, blob_path: str, local_path: str) -> str:
        client = Client()
        bucket = client.get_bucket(self.BUCKET)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        return local_path

    def _upload_to_cloudstorage(self, local_path: str, blob_path: str) -> str:
        client = Client()
        bucket = client.bucket(self.BUCKET)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(local_path, timeout=3600)
        return blob_path

    def _remove_from_cloudstorage(self, blob_path: str):
        client = Client()
        bucket = client.bucket(self.BUCKET)
        try:  # don't fail entire task if this fails
            bucket.delete_blob(blob_path)
        except NotFound:
            print(f"{blob_path} not found")

    def _parse_task_id(self, output: Union[str, bytes]) -> Optional[str]:
        if isinstance(output, bytes):
            text = output.decode("utf-8")
        else:
            text = output

        task_id_regex = re.compile(r"(?<=ID: ).*", flags=re.IGNORECASE)
        try:
            matches = task_id_regex.search(text)
            if matches is None:
                return None
            return matches[0]
        except TypeError:
            return None

    def _cp_storage_to_ee_table(
        self, blob_uri: str, table_asset_id: str, geofield: str = "geom"
    ) -> str:
        try:
            cmd = [
                "/usr/local/bin/earthengine",
                f"--service_account_file={self.google_creds_path}",
                "upload table",
                f"--primary_geometry_column {geofield}",
                f"--asset_id={table_asset_id}",
                blob_uri,
            ]
            output = subprocess.check_output(
                " ".join(cmd), stderr=subprocess.STDOUT, shell=True
            )
            task_id = self._parse_task_id(output)
            if task_id is None:
                raise TypeError("task_id is None")
            self.ee_tasks[task_id] = {}
            return task_id
        except subprocess.CalledProcessError as err:
            raise ConversionException(err.stdout)

    def _get_df(self, query):
        _scenario_clause = (
            f"AND ScenarioName IS NULL OR ScenarioName = '{self.CANONICAL}'"
        )
        if self.scenario and self.scenario != self.CANONICAL:
            _scenario_clause = f"AND ScenarioName = '{self.scenario}'"

        query = f"{query} {_scenario_clause}"
        obsconn = pyodbc.connect(self._obsconn_str)
        df = pd.read_sql(query, obsconn)
        return df

    # add "mastergrid" and "mastergridcell" to df
    def zonify(self, df):
        master_grid_df = pd.DataFrame(columns=self.ZONIFY_DF_COLUMNS)

        # df with point geom if we have one, polygon otherwise, or drop if neither
        obs_df = df[
            [self.POINT_LOC_LABEL, self.GRIDCELL_LOC_LABEL, self.UNIQUE_ID_LABEL]
        ]
        obs_df = obs_df.dropna(
            subset=[self.POINT_LOC_LABEL, self.GRIDCELL_LOC_LABEL], how="all"
        )
        obs_df = obs_df.assign(geom=obs_df[self.POINT_LOC_LABEL])
        obs_df["geom"].loc[obs_df["geom"].isna()] = obs_df[self.GRIDCELL_LOC_LABEL]
        obs_df = obs_df[["geom", self.UNIQUE_ID_LABEL]]
        obs_df.set_index(self.UNIQUE_ID_LABEL, inplace=True)

        if not obs_df.empty:
            obs_features = self.df2fc(obs_df).filterBounds(self.geofilter)
            return_obs_features = obs_features.map(self.attribute_obs)
            master_grid_df = self.fc2df(return_obs_features, self.ZONIFY_DF_COLUMNS)

        df = pd.merge(left=df, right=master_grid_df)

        # save out non-intersecting observations
        df_nonintersections = df[
            (df[self.MASTER_GRID_LABEL] == self.EE_NODATA)
            | (df[self.MASTER_CELL_LABEL] == self.EE_NODATA)
            | (df[self.SCLPOLY_ID] == self.EE_NODATA)
        ]
        if not df_nonintersections.empty:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            df_nonintersections.to_csv(f"nonintersecting-{timestr}.csv")

        # Filter out rows not in any zone and rows not in any gridcell (-9999)
        df = df[
            (df[self.MASTER_GRID_LABEL] != self.EE_NODATA)
            & (df[self.MASTER_CELL_LABEL] != self.EE_NODATA)
            & (df[self.SCLPOLY_ID] != self.EE_NODATA)
        ]

        return df

    def fc2df(self, featurecollection, columns=None):
        df = pd.DataFrame()
        fc_exists = False
        try:
            fc_exists = featurecollection.first().getInfo()  # exists and is not empty
        except ee.ee_exception.EEException as e:
            pass

        if fc_exists:
            tempfile = str(uuid.uuid4())
            blob = f"prob/{self.species}/{self.scenario}/{self.taskdate}/{tempfile}"
            task_id = self.export_fc_cloudstorage(
                featurecollection, self.BUCKET, blob, "CSV", columns
            )
            self.wait()
            csv = self._download_from_cloudstorage(f"{blob}.csv", f"{tempfile}.csv")
            self.fc_csvs.append((f"{tempfile}.csv", None))

            # uncomment to export shp for QA
            # shp_task_id = self.export_fc_cloudstorage(
            #     featurecollection, self.BUCKET, blob, "SHP", columns
            # )

            df = pd.read_csv(csv, encoding="utf-8")
            self._remove_from_cloudstorage(f"{blob}.csv")
        return df

    def df2fc(
        self, df: pd.DataFrame, geofield: str = "geom"
    ) -> Optional[ee.FeatureCollection]:
        tempfile = str(uuid.uuid4())
        blob = f"prob/{self.species}/{self.scenario}/{self.taskdate}/{tempfile}"
        if df.empty:
            return None

        df.replace(np.inf, 0, inplace=True)
        df.to_csv(f"{tempfile}.csv")
        self._upload_to_cloudstorage(f"{tempfile}.csv", f"{blob}.csv")
        table_asset_name, table_asset_id = self._prep_asset_id(f"scratch/{tempfile}")
        task_id = self._cp_storage_to_ee_table(
            f"gs://{self.BUCKET}/{blob}.csv", table_asset_id, geofield
        )
        self.wait()
        self._remove_from_cloudstorage(f"{blob}.csv")
        self.fc_csvs.append((f"{tempfile}.csv", table_asset_id))
        return ee.FeatureCollection(table_asset_id)

    @property
    def df_adhoc(self):
        if self._df_adhoc is None:
            _csvpath = "adhoc.csv"
            if self.use_cache and Path(_csvpath).is_file():
                self._df_adhoc = pd.read_csv(
                    _csvpath, encoding="utf-8", index_col=self.MASTER_CELL_LABEL
                )
            else:
                query = (
                    f"SELECT {self.UNIQUE_ID_LABEL}, "
                    f"{self.POINT_LOC_LABEL}, "
                    f"{self.GRIDCELL_LOC_LABEL}, "
                    f"EndObservationDate AS {self.DATE_LABEL}, "
                    f"{self.DENSITY_LABEL} "
                    f"FROM dbo.vw_CI_AdHocObservation "
                    f"WHERE DATEDIFF(YEAR, EndObservationDate, '{self.taskdate}') <= {self.inputs['obs_adhoc']['maxage']} "
                    f"AND EndObservationDate <= Cast('{self.taskdate}' AS datetime) "
                )
                self._df_adhoc = self._get_df(query)
                print("zonify adhoc")
                self._df_adhoc = self.zonify(self._df_adhoc)
                self._df_adhoc = self._df_adhoc.drop(
                    [self.POINT_LOC_LABEL, self.GRIDCELL_LOC_LABEL], axis=1
                )
                self._df_adhoc.set_index(self.MASTER_CELL_LABEL, inplace=True)

            if self.use_cache:
                self._df_adhoc.to_csv(_csvpath)

        return self._df_adhoc

    @property
    def df_cameratrap(self):
        if self._df_ct is None:
            _csvpath = "cameratrap.csv"
            if self.use_cache and Path(_csvpath).is_file():
                self._df_ct = pd.read_csv(
                    _csvpath, encoding="utf-8", index_col="CameraTrapDeploymentID"
                )
            else:
                query = (
                    f"SELECT {self.UNIQUE_ID_LABEL}, "
                    f"CameraTrapDeploymentID, "
                    f"{self.POINT_LOC_LABEL}, "
                    f"{self.GRIDCELL_LOC_LABEL}, "
                    f"PickupDatetime AS {self.DATE_LABEL}, "
                    f"{self.CT_DAYS_OPERATING_LABEL} "
                    f"FROM dbo.vw_CI_CameraTrapDeployment "
                    f"WHERE DATEDIFF(YEAR, PickupDatetime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                    f"AND PickupDatetime <= Cast('{self.taskdate}' AS datetime) "
                )
                df_ct_dep = self._get_df(query)
                print("zonify camera trap deployments")
                df_ct_dep = self.zonify(df_ct_dep)
                df_ct_dep.set_index("CameraTrapDeploymentID", inplace=True)

                query = (
                    f"SELECT * FROM dbo.vw_CI_CameraTrapObservation "
                    f"WHERE DATEDIFF(YEAR, ObservationDate, '{self.taskdate}') <= "
                    f"{self.inputs['obs_ct']['maxage']} "
                    f"AND ObservationDate <= Cast('{self.taskdate}' AS date) "
                )
                df_ct_obs = self._get_df(query)

                df_ct_obs.set_index("CameraTrapDeploymentID", inplace=True)
                df_ct_obs[self.CT_DAYS_DETECTED_LABEL] = (
                    df_ct_obs["AdultMaleCount"]
                    + df_ct_obs["AdultFemaleCount"]
                    + df_ct_obs["AdultSexUnknownCount"]
                    + df_ct_obs["SubAdultCount"]
                    + df_ct_obs["YoungCount"]
                )
                df_ct_obs[self.CT_DAYS_DETECTED_LABEL] = df_ct_obs[
                    self.CT_DAYS_DETECTED_LABEL
                ].astype(int)

                # number of days with > 0 detections per camera
                df_ct_obsdays = (
                    df_ct_obs[df_ct_obs[self.CT_DAYS_DETECTED_LABEL] > 0]
                    .groupby(["CameraTrapDeploymentID"])
                    .count()
                )
                self._df_ct = pd.merge(
                    left=df_ct_dep,
                    right=df_ct_obsdays,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
                self._df_ct[self.CT_DAYS_DETECTED_LABEL].fillna(0, inplace=True)
                self._df_ct.rename(
                    columns={"UniqueID_x": self.UNIQUE_ID_LABEL}, inplace=True
                )
                self._df_ct = self._df_ct[
                    [
                        self.UNIQUE_ID_LABEL,
                        self.DATE_LABEL,
                        self.MASTER_GRID_LABEL,
                        self.MASTER_CELL_LABEL,
                        self.SCLPOLY_ID,
                        self.CT_DAYS_OPERATING_LABEL,
                        self.CT_DAYS_DETECTED_LABEL,
                    ]
                ]

            if self.use_cache:
                self._df_ct.to_csv(_csvpath)

        return self._df_ct

    @property
    def df_signsurvey(self):
        if self._df_ss is None:
            _csvpath = "signsurvey.csv"
            if self.use_cache and Path(_csvpath).is_file():
                self._df_ss = pd.read_csv(
                    _csvpath, encoding="utf-8", index_col=self.MASTER_CELL_LABEL
                )
            else:
                query = (
                    f"SELECT {self.UNIQUE_ID_LABEL}, "
                    f"{self.POINT_LOC_LABEL}, "
                    f"{self.GRIDCELL_LOC_LABEL}, "
                    f"StartDate AS {self.DATE_LABEL}, "
                    f"{self.SS_SEGMENTS_SURVEYED_LABEL}, "
                    f"{self.SS_SEGMENTS_DETECTED_LABEL} "
                    f"FROM dbo.vw_CI_SignSurveyObservation "
                    f"WHERE DATEDIFF(YEAR, StartDate, '{self.taskdate}') <= {self.inputs['obs_ss']['maxage']} "
                    f"AND StartDate <= Cast('{self.taskdate}' AS datetime) "
                )
                self._df_ss = self._get_df(query)
                print("zonify sign survey")
                self._df_ss = self.zonify(self._df_ss)
                self._df_ss = self._df_ss.drop(
                    [self.POINT_LOC_LABEL, self.GRIDCELL_LOC_LABEL], axis=1
                )
                self._df_ss.set_index(self.MASTER_CELL_LABEL, inplace=True)

            if self.use_cache:
                self._df_ss.to_csv(_csvpath)

        return self._df_ss

    @property
    def scl_polys(self):
        return self.scl.map(self.attribute_scl)

    def attribute_scl(self, scl):
        # TODO: take country/biome from intersected polygon with the largest area
        poly = scl.geometry()

        matching_countries = ee.Join.simple().apply(
            self.countries.filterBounds(poly), scl, self.intersects
        )
        country_true = matching_countries.first().get("COUNTRY_NA")
        country = ee.Algorithms.If(
            matching_countries.size().gte(1), country_true, ee.String("")
        )

        matching_ecoregions = ee.Join.simple().apply(
            self.ecoregions.filterBounds(poly), scl, self.intersects
        )
        biome_true = matching_ecoregions.first().get("BIOME_NUM")
        biome = ee.Algorithms.If(
            matching_ecoregions.size().gte(1), biome_true, ee.Number(self.EE_NODATA)
        )

        range = self.range_class.reduceRegion(
            geometry=poly, reducer=ee.Reducer.mode(), scale=self.scale, crs=self.crs
        )

        return scl.set(
            {
                self.COUNTRY_LABEL: country,
                self.BIOME_LABEL: biome,
                self.RANGE_LABEL: range,
            }
        )

    def attribute_obs(self, obs_feature):
        centroid = obs_feature.centroid().geometry()

        matching_zones = ee.Join.simple().apply(
            self.zones, obs_feature, self.intersects
        )
        zone_id_true = ee.Number(matching_zones.first().get(self.ZONES_LABEL))
        id_false = ee.Number(self.EE_NODATA)
        zone_id = ee.Number(
            ee.Algorithms.If(matching_zones.size().gte(1), zone_id_true, id_false)
        )

        matching_gridcells = self.gridcells.filter(
            ee.Filter.eq("zone", zone_id)
        ).filterBounds(centroid)
        gridcell_id_true = ee.Number(matching_gridcells.first().get(self.EE_ID_LABEL))
        gridcell_id = ee.Number(
            ee.Algorithms.If(
                zone_id.neq(self.EE_NODATA),
                ee.Algorithms.If(
                    matching_gridcells.size().gte(1), gridcell_id_true, id_false
                ),
                id_false,
            )
        )

        matching_polys = self.scl_polys.filterBounds(centroid)
        poly_true = ee.Number(matching_polys.first().get(self.EE_SYSTEM_INDEX))
        poly = ee.Algorithms.If(matching_polys.size().gte(1), poly_true, id_false)

        obs_feature = obs_feature.setMulti(
            {
                self.MASTER_GRID_LABEL: zone_id,
                self.MASTER_CELL_LABEL: gridcell_id,
                self.SCLPOLY_ID: poly,
            }
        )

        return obs_feature

    def calc(self):
        # Temporary: export dataframes needed for probability modeling
        # Make sure cache csvs don't exist locally before running
        if self.use_cache:
            prob_columns = [
                "system:index",
                "biome",
                "country",
            ]
            df_scl_polys = self.fc2df(self.scl_polys, columns=prob_columns)
            df_scl_polys.rename(columns={"system:index": self.SCLPOLY_ID}, inplace=True)
            df_scl_polys.set_index(self.SCLPOLY_ID, inplace=True)
            df_scl_polys.to_csv("scl_polys.csv")

        print(self.df_adhoc)
        print(self.df_cameratrap)
        print(self.df_signsurvey)

        # temporary code here to:
        # convert polys csv Charles sends to fc
        # scl_scored = self.scl_polys mapped to transfer probability/effort results

        # scl_species = scl_scored.filter(self.scl_poly_filters["scl_species"])
        # scl_survey = scl_scored.filter(self.scl_poly_filters["scl_survey"])
        # scl_restoration = scl_scored.filter(self.scl_poly_filters["scl_restoration"])
        # scl_fragment = scl_scored.filter(self.scl_poly_filters["scl_fragment"])
        #
        # self.poly_export(scl_species, "scl_species")
        # self.poly_export(scl_survey, "scl_survey")
        # self.poly_export(scl_restoration, "scl_restoration")
        # self.poly_export(scl_fragment, "scl_fragment")

    def check_inputs(self):
        super().check_inputs()

    def clean_up(self, **kwargs):
        if self.status == self.FAILED:
            return

        if self.fc_csvs:
            for csv, table_asset_id in self.fc_csvs:
                if csv and Path(csv).exists():
                    Path(csv).unlink()
                if table_asset_id:
                    try:
                        asset = ee.data.getAsset(table_asset_id)
                        ee.data.deleteAsset(table_asset_id)
                    except ee.ee_exception.EEException:
                        print(f"{table_asset_id} does not exist; skipping")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--taskdate")
    parser.add_argument("-s", "--species")
    parser.add_argument("-u", "--use-cache", action="store_true")
    parser.add_argument("--scenario")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing outputs instead of incrementing",
    )
    options = parser.parse_args()
    sclclassification_task = SCLClassification(**vars(options))
    sclclassification_task.run()
