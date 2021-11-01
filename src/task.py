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
from task_base import SCLTask, EETaskError


class ConversionException(Exception):
    pass


# noinspection PyTypeChecker
class SCLClassification(SCLTask):
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
    PROBABILITY = "phi_3"
    EFFORT = "eff_3"
    RANGE = "range"

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
        "scl_image": {
            "ee_type": SCLTask.IMAGECOLLECTION,
            "ee_path": "scl_image_path",
            "maxage": 1 / 365,
        },
        "zones": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "zones_path",
            "static": True,
        },
        "gridcells": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "gridcells_path",
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
        "current_range": 2,
        "probability": 0.99,
        "survey_effort": 0.6,
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
        self.scl_image, _ = self.get_most_recent_image(
            ee.ImageCollection(self.inputs["scl_image"]["ee_path"])
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
        self.biomes = self.ecoregions.reduceToImage(["BIOME_NUM"], ee.Reducer.mode())
        self.intersects = ee.Filter.intersects(".geo", None, ".geo")

        self.scl_poly_filters = {
            "scl_species": ee.Filter.And(
                ee.Filter.greaterThanOrEquals(
                    leftField=self.CONNECTED_HABITAT_AREA, rightField=self.MIN_PATCHSIZE
                ),
                ee.Filter.eq(self.RANGE, self.thresholds["current_range"]),
                ee.Filter.gte(self.PROBABILITY, self.thresholds["probability"]),
            ),
            "scl_restoration": ee.Filter.Or(
                ee.Filter.And(
                    ee.Filter.greaterThanOrEquals(
                        leftField=self.CONNECTED_HABITAT_AREA,
                        rightField=self.MIN_PATCHSIZE,
                    ),
                    ee.Filter.eq(self.RANGE, self.thresholds["current_range"]),
                    ee.Filter.lt(self.PROBABILITY, self.thresholds["probability"]),
                    ee.Filter.gte(self.EFFORT, self.thresholds["survey_effort"]),
                ),
                ee.Filter.And(
                    ee.Filter.greaterThanOrEquals(
                        leftField=self.CONNECTED_HABITAT_AREA,
                        rightField=self.MIN_PATCHSIZE,
                    ),
                    ee.Filter.lt(self.RANGE, self.thresholds["current_range"]),
                ),
            ),
            "scl_survey": ee.Filter.And(
                ee.Filter.greaterThanOrEquals(
                    leftField=self.CONNECTED_HABITAT_AREA, rightField=self.MIN_PATCHSIZE
                ),
                ee.Filter.eq(self.RANGE, self.thresholds["current_range"]),
                ee.Filter.lt(self.PROBABILITY, self.thresholds["probability"]),
                ee.Filter.lt(  # you haven't put in enough effort to lower the probability a critter is present
                    self.EFFORT, self.thresholds["survey_effort"]
                ),
            ),
            "scl_fragment_historical_presence": ee.Filter.And(
                ee.Filter.lessThan(
                    leftField=self.CONNECTED_HABITAT_AREA, rightField=self.MIN_PATCHSIZE
                ),
                ee.Filter.eq(self.RANGE, self.thresholds["current_range"]),
                ee.Filter.gte(self.PROBABILITY, self.thresholds["probability"]),
            ),
            "scl_fragment_historical_nopresence": ee.Filter.And(
                ee.Filter.lessThan(
                    leftField=self.CONNECTED_HABITAT_AREA, rightField=self.MIN_PATCHSIZE
                ),
                ee.Filter.eq(self.RANGE, self.thresholds["current_range"]),
                ee.Filter.lt(self.PROBABILITY, self.thresholds["probability"]),
            ),
            "scl_fragment_extirpated_presence": ee.Filter.And(
                ee.Filter.lessThan(
                    leftField=self.CONNECTED_HABITAT_AREA, rightField=self.MIN_PATCHSIZE
                ),
                ee.Filter.lt(self.RANGE, self.thresholds["current_range"]),
                ee.Filter.gte(self.PROBABILITY, self.thresholds["probability"]),
            ),
            "scl_fragment_extirpated_nopresence": ee.Filter.And(
                ee.Filter.lessThan(
                    leftField=self.CONNECTED_HABITAT_AREA, rightField=self.MIN_PATCHSIZE
                ),
                ee.Filter.lt(self.RANGE, self.thresholds["current_range"]),
                ee.Filter.lt(self.PROBABILITY, self.thresholds["probability"]),
            ),
        }

    def scl_polys_path(self):
        return f"{self.ee_rootdir}/pothab/scl_polys"

    def scl_image_path(self):
        return f"{self.ee_rootdir}/pothab/scl_image"

    def zones_path(self):
        return f"{self.speciesdir}/zones"

    def gridcells_path(self):
        return f"{self.speciesdir}/gridcells"

    def poly_export(self, polys, scl_name):
        path = f"pothab/{scl_name}"
        self.export_fc_ee(polys, path)

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

    def _cp_storage_to_ee_table(self, blob_uri: str, table_asset_id: str) -> str:
        try:
            cmd = [
                "/usr/local/bin/earthengine",
                f"--service_account_file={self.google_creds_path}",
                "upload table",
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

    def _prep_obs_df(self, df):
        # df with point geom if we have one, polygon otherwise, or drop if neither
        obs_df = df[[self.POINT_LOC, self.GRIDCELL_LOC, self.UNIQUE_ID]]
        obs_df = obs_df.dropna(subset=[self.POINT_LOC, self.GRIDCELL_LOC], how="all")
        obs_df = obs_df.assign(geom=obs_df[self.POINT_LOC])
        obs_df["geom"].loc[obs_df["geom"].isna()] = obs_df[self.GRIDCELL_LOC]
        obs_df = obs_df[["geom", self.UNIQUE_ID]]
        obs_df.set_index(self.UNIQUE_ID, inplace=True)
        return obs_df

    def inner_join(self, primary, secondary, primary_field, secondary_field):
        def _flatten_fields(feat):
            primary_feature = ee.Feature(feat.get("primary"))
            secondary_feature = ee.Feature(feat.get("secondary"))
            return_feat = ee.Feature(
                primary_feature.geometry(),
                primary_feature.toDictionary().combine(
                    secondary_feature.toDictionary()
                ),
            )
            return return_feat

        return (
            ee.Join.inner("primary", "secondary").apply(
                primary,
                secondary,
                ee.Filter.equals(leftField=primary_field, rightField=secondary_field),
            )
        ).map(_flatten_fields)

    def zonify(self, df):
        master_grid_df = pd.DataFrame(columns=self.ZONIFY_DF_COLUMNS)

        def _max_frequency(feat):
            hist = ee.Dictionary(feat.get(self.MASTER_CELL))
            vals = hist.values()
            max = vals.reduce(ee.Reducer.max())
            index = vals.indexOf(max)
            max_key = ee.Number.parse(hist.keys().get(index)).toLong()
            return feat.set(self.MASTER_CELL, max_key)

        obs_df = self._prep_obs_df(df)
        if not obs_df.empty:
            obs_features = self.df2fc(obs_df).filterBounds(self.geofilter)

            scl_image = self.scl_polys.reduceToImage(
                [self.SCLPOLY_ID], ee.Reducer.mode()
            )
            gridcells = scl_image.reduceRegions(
                collection=self.gridcells,
                reducer=ee.Reducer.mode().setOutputs([self.SCLPOLY_ID]),
                scale=self.scale,
                crs=self.crs,
            )
            gridcellimage = (
                ee.ImageCollection(
                    [
                        gridcells.reduceToImage([self.ZONES], ee.Reducer.mode()),
                        gridcells.reduceToImage([self.EE_ID_LABEL], ee.Reducer.mode()),
                    ]
                )
                .toBands()
                .rename([self.MASTER_GRID, self.MASTER_CELL])
            )

            gridded_obs_features = gridcellimage.reduceRegions(
                collection=obs_features,
                reducer=ee.Reducer.mode()
                .forEach([self.MASTER_GRID])
                .combine(ee.Reducer.frequencyHistogram().forEach([self.MASTER_CELL])),
                scale=self.scale,
                crs=self.crs,
            ).map(_max_frequency)

            return_obs_features = self.inner_join(
                gridded_obs_features, gridcells, self.MASTER_CELL, self.EE_ID_LABEL
            )

            # self.export_fc_ee(return_obs_features, "scratch/signsurvey_211029")
            master_grid_df = self.fc2df(return_obs_features, self.ZONIFY_DF_COLUMNS)

        df = pd.merge(left=df, right=master_grid_df)

        # save out non-intersecting observations
        # df_nonintersections = df[
        #     (df[self.MASTER_GRID_LABEL] == self.EE_NODATA)
        #     | (df[self.MASTER_CELL_LABEL] == self.EE_NODATA)
        #     | (df[self.SCLPOLY_ID] == self.EE_NODATA)
        # ]
        # if not df_nonintersections.empty:
        #     timestr = time.strftime("%Y%m%d-%H%M%S")
        #     df_nonintersections.to_csv(f"nonintersecting-{timestr}.csv")
        #
        # # Filter out rows not in any zone and rows not in any gridcell (-9999)
        # df = df[
        #     (df[self.MASTER_GRID_LABEL] != self.EE_NODATA)
        #     & (df[self.MASTER_CELL_LABEL] != self.EE_NODATA)
        #     & (df[self.SCLPOLY_ID] != self.EE_NODATA)
        # ]

        return df

    def fc2df(self, featurecollection, columns=None):
        tempfile = str(uuid.uuid4())
        blob = f"prob/{self.species}/{self.scenario}/{self.taskdate}/{tempfile}"
        task_id = self.export_fc_cloudstorage(
            featurecollection, self.BUCKET, blob, "CSV", columns
        )
        self.wait()
        csv = self._download_from_cloudstorage(f"{blob}.csv", f"{tempfile}.csv")
        self.fc_csvs.append((f"{tempfile}.csv", None))

        # uncomment to export for QA in a GIS
        # shp_task_id = self.export_fc_cloudstorage(
        #     featurecollection, self.BUCKET, blob, "GeoJSON", columns
        # )

        df = pd.read_csv(csv, encoding="utf-8")
        self._remove_from_cloudstorage(f"{blob}.csv")
        return df

    def df2fc(self, df: pd.DataFrame) -> Optional[ee.FeatureCollection]:
        tempfile = str(uuid.uuid4())
        blob = f"prob/{self.species}/{self.scenario}/{self.taskdate}/{tempfile}"
        if df.empty:
            return None

        df.replace(np.inf, 0, inplace=True)
        df.to_csv(f"{tempfile}.csv")
        self._upload_to_cloudstorage(f"{tempfile}.csv", f"{blob}.csv")
        table_asset_name, table_asset_id = self._prep_asset_id(f"scratch/{tempfile}")
        task_id = self._cp_storage_to_ee_table(
            f"gs://{self.BUCKET}/{blob}.csv", table_asset_id
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
                    _csvpath, encoding="utf-8", index_col=self.MASTER_CELL
                )
            else:
                query = (
                    f"SELECT {self.UNIQUE_ID}, "
                    f"{self.POINT_LOC}, "
                    f"{self.GRIDCELL_LOC}, "
                    f"EndObservationDate AS {self.DATE}, "
                    f"{self.DENSITY} "
                    f"FROM dbo.vw_CI_AdHocObservation "
                    f"WHERE DATEDIFF(YEAR, EndObservationDate, '{self.taskdate}') <= {self.inputs['obs_adhoc']['maxage']} "
                    f"AND EndObservationDate <= Cast('{self.taskdate}' AS datetime) "
                )
                self._df_adhoc = self._get_df(query)
                print("zonify adhoc")
                self._df_adhoc = self.zonify(self._df_adhoc)
                self._df_adhoc = self._df_adhoc.drop(
                    [self.POINT_LOC, self.GRIDCELL_LOC], axis=1
                )
                self._df_adhoc.set_index(self.MASTER_CELL, inplace=True)

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
                    f"SELECT {self.UNIQUE_ID}, "
                    f"CameraTrapDeploymentID, "
                    f"{self.POINT_LOC}, "
                    f"{self.GRIDCELL_LOC}, "
                    f"PickupDatetime AS {self.DATE}, "
                    f"{self.CT_DAYS_OPERATING} "
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
                df_ct_obs[self.CT_DAYS_DETECTED] = (
                    df_ct_obs["AdultMaleCount"]
                    + df_ct_obs["AdultFemaleCount"]
                    + df_ct_obs["AdultSexUnknownCount"]
                    + df_ct_obs["SubAdultCount"]
                    + df_ct_obs["YoungCount"]
                )
                df_ct_obs[self.CT_DAYS_DETECTED] = df_ct_obs[
                    self.CT_DAYS_DETECTED
                ].astype(int)

                # number of days with > 0 detections per camera
                df_ct_obsdays = (
                    df_ct_obs[df_ct_obs[self.CT_DAYS_DETECTED] > 0]
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
                self._df_ct[self.CT_DAYS_DETECTED].fillna(0, inplace=True)
                self._df_ct.rename(columns={"UniqueID_x": self.UNIQUE_ID}, inplace=True)
                self._df_ct = self._df_ct[
                    [
                        self.UNIQUE_ID,
                        self.DATE,
                        self.MASTER_GRID,
                        self.MASTER_CELL,
                        self.SCLPOLY_ID,
                        self.CT_DAYS_OPERATING,
                        self.CT_DAYS_DETECTED,
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
                    _csvpath, encoding="utf-8", index_col=self.MASTER_CELL
                )
            else:
                query = (
                    f"SELECT {self.UNIQUE_ID}, "
                    f"{self.POINT_LOC}, "
                    f"{self.GRIDCELL_LOC}, "
                    f"StartDate AS {self.DATE}, "
                    f"{self.SS_SEGMENTS_SURVEYED}, "
                    f"{self.SS_SEGMENTS_DETECTED} "
                    f"FROM dbo.vw_CI_SignSurveyObservation "
                    f"WHERE DATEDIFF(YEAR, StartDate, '{self.taskdate}') <= {self.inputs['obs_ss']['maxage']} "
                    f"AND StartDate <= Cast('{self.taskdate}' AS datetime) "
                )
                self._df_ss = self._get_df(query)
                print("zonify sign survey")
                self._df_ss = self.zonify(self._df_ss)
                self._df_ss = self._df_ss.drop(
                    [self.POINT_LOC, self.GRIDCELL_LOC], axis=1
                )
                self._df_ss.set_index(self.MASTER_CELL, inplace=True)

            if self.use_cache:
                self._df_ss.to_csv(_csvpath)

        return self._df_ss

    @property
    def scl_polys(self):
        def _assign_ids(item):
            item = ee.List(item)
            feature = ee.Feature(item.get(0))
            poly_id = item.get(1)
            return feature.set({self.SCLPOLY_ID: poly_id})

        ids = ee.List.sequence(1, self.scl.size())
        scl_poly_list = ee.List(self.scl.toList(self.scl.size()))
        scl_polys_assigned = ee.FeatureCollection(
            scl_poly_list.zip(ids).map(_assign_ids)
        )

        return self.biomes.reduceRegions(
            collection=scl_polys_assigned,
            reducer=ee.Reducer.mode().setOutputs([self.BIOME]),
            scale=self.scale,
            crs=self.crs,
        )

    def dissolve(self, polys, core_label, fragment_label):
        cores = polys.filter(self.scl_poly_filters[core_label])
        fragments = polys.filter(self.scl_poly_filters[fragment_label])

        def _item_to_classified_feature(item):
            geom = ee.Geometry.Polygon(item)
            contained = cores.geometry().intersects(geom)
            lstype = ee.Algorithms.If(contained, 1, 2)
            return ee.Feature(geom, {"lstype": lstype})

        dissolved_list = cores.merge(fragments).geometry().dissolve().coordinates()
        dissolved_polys = (
            ee.FeatureCollection(dissolved_list.map(_item_to_classified_feature))
            .reduceToImage(["lstype"], ee.Reducer.first())
            .setDefaultProjection(
                scale=450,
                crs=self.crs,  # this could probably be anything <500
            )
            .unmask(0)
            .reduceResolution(ee.Reducer.min())
            .reproject(
                scale=self.scale,
                crs=self.crs,
            )
            .selfMask()
            .reduceToVectors(
                geometry=ee.Geometry.Polygon(self.extent),
                crs=self.crs,
                scale=self.scale,
                labelProperty="lstype",
                maxPixels=self.ee_max_pixels,
            )
        )

        dissolved_cores = dissolved_polys.filter(ee.Filter.eq("lstype", 1))
        dissolved_fragments = dissolved_polys.filter(ee.Filter.eq("lstype", 2))

        return dissolved_cores, dissolved_fragments

    def reattribute(self, polys):
        def _round(feat):
            geom = feat.geometry()
            eph = feat.get(self.HABITAT_AREA)
            ceph = feat.get(self.CONNECTED_HABITAT_AREA)
            return ee.Feature(
                geom,
                {
                    self.HABITAT_AREA: ee.Number(eph).round(),
                    self.CONNECTED_HABITAT_AREA: ee.Number(ceph).round(),
                },
            )

        return (
            self.scl_image.select(
                [self.HABITAT_AREA, self.CONNECTED_HABITAT_AREA]
            ).reduceRegions(
                collection=polys,
                reducer=ee.Reducer.sum(),
                scale=self.scale,
                crs=self.crs,
            )
        ).map(_round)

    def calc(self):
        # Temporary: export dataframes needed for probability modeling
        # Make sure cache csvs don't exist locally before running
        if self.use_cache:
            prob_columns = [
                self.SCLPOLY_ID,
                self.BIOME,
                self.COUNTRY,
            ]

            # for join debugging
            self.export_fc_ee(self.scl_polys, "assigned_scl_polys")

            df_scl_polys = self.fc2df(self.scl_polys, columns=prob_columns)
            df_scl_polys.set_index(self.SCLPOLY_ID, inplace=True)
            print(df_scl_polys)
            df_scl_polys.to_csv("scl_polys.csv")

        print(self.df_adhoc)
        print(self.df_cameratrap)
        print(self.df_signsurvey)

        # Temporary: ingest probability model output csv, join to polys, and do classification
        probout = self.df2fc(
            pd.read_csv(
                "scl_polys_out.csv", encoding="utf-8", index_col=self.SCLPOLY_ID
            ),
        )

        scl_scored = self.inner_join(
            self.scl_polys, probout, self.SCLPOLY_ID, self.SCLPOLY_ID
        )

        scl_species, scl_species_fragment = self.dissolve(
            scl_scored, "scl_species", "scl_fragment_historical_presence"
        )
        scl_survey, scl_survey_fragment = self.dissolve(
            scl_scored, "scl_survey", "scl_fragment_historical_nopresence"
        )
        scl_restoration, scl_rest_frag = self.dissolve(
            scl_scored, "scl_restoration", "scl_fragment_extirpated_presence"
        )

        self.poly_export(self.reattribute(scl_species), "scl_species")
        self.poly_export(self.reattribute(scl_species_fragment), "scl_species_fragment")
        self.poly_export(self.reattribute(scl_survey), "scl_survey")
        self.poly_export(self.reattribute(scl_survey_fragment), "scl_survey_fragment")
        self.poly_export(self.reattribute(scl_restoration), "scl_restoration")
        self.poly_export(self.reattribute(scl_rest_frag), "scl_restoration_fragment")

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

    try:
        sclclassification_task.run()
    except EETaskError as e:
        statuses = list(e.ee_statuses.values())
        if statuses:
            message = statuses[0]["error_message"]
            if message.lower() == "table is empty.":
                sclclassification_task.status = sclclassification_task.RUNNING
            else:
                raise e
