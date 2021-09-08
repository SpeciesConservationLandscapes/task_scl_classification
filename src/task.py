import argparse
import ee
import os
import re
import subprocess
import numpy as np
import pandas as pd
import pyodbc

# import time
import uuid

# import json
# from datetime import datetime, timezone
from typing import Optional, Union

# from geomet import wkt
from google.cloud.storage import Client
from google.cloud.exceptions import NotFound
from pathlib import Path
from task_base import SCLTask


# noinspection PyTypeChecker,PyUnresolvedReferences
class SCLClassification(SCLTask):
    BUCKET = "scl-pipeline"
    POINT_LOC_LABEL = "PointLocation"
    GRIDCELL_LOC_LABEL = "GridCellLocation"
    UNIQUE_ID_LABEL = "UniqueID"
    DATE_LABEL = "datestamp"
    OBS_DECAY_RATE = -0.001  # for use with age in days

    google_creds_path = "/.google_creds"
    inputs = {
        "obs_adhoc": {"maxage": 50},
        "obs_ss": {"maxage": 50},
        "obs_ct": {"maxage": 50},
        # "zones": {
        #     "ee_type": SCLTask.FEATURECOLLECTION,
        #     "ee_path": "scl_zones",
        #     "static": True,
        # },
        "scl": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "scl_polys",
            "maxage": 1 / 365,
        },
    }
    thresholds = {
        "landscape_size": 3,
        "current_range": 2,
        "presence_score": 1,
        "landscape_survey_effort": 1,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        self._fc_observations = None
        # self._zone_ids = []
        # self.zone = None
        self.fc_csvs = []
        # self.zones = ee.FeatureCollection(self.inputs["zones"]["ee_path"])
        self.scl, _ = self.get_most_recent_featurecollection(
            self.inputs["scl"]["ee_path"]
        )
        self.intersects = ee.Filter.intersects(".geo", None, ".geo")

        self.scl_poly_filters = {
            "scl_species": ee.Filter.And(
                ee.Filter.gte("size", self.thresholds["landscape_size"]),
                ee.Filter.eq("range", self.thresholds["current_range"]),
                ee.Filter.gte("presence_score", self.thresholds["presence_score"]),
                ee.Filter.gte("effort", self.thresholds["landscape_survey_effort"]),
            ),
            "scl_restoration": ee.Filter.Or(
                ee.Filter.And(
                    ee.Filter.gte("size", self.thresholds["landscape_size"]),
                    ee.Filter.eq("range", self.thresholds["current_range"]),
                    ee.Filter.lt("presence_score", self.thresholds["presence_score"]),
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
                ee.Filter.gte("presence_score", self.thresholds["presence_score"]),
                ee.Filter.lt("effort", self.thresholds["landscape_survey_effort"]),
            ),
            "scl_fragment": ee.Filter.And(
                ee.Filter.lt("size", self.thresholds["landscape_size"]),
                ee.Filter.eq("range", self.thresholds["current_range"]),
                ee.Filter.gte("presence_score", self.thresholds["presence_score"]),
                ee.Filter.gte("effort", self.thresholds["landscape_survey_effort"]),
            ),
        }

    # TODO: move to task_base
    # def scl_zones(self):
    #     return f"projects/{self.ee_project}/{self.species}/zones"

    def scl_polys(self):
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

    def _obs_score(self, row):
        age = self.taskdate - row[self.DATE_LABEL].to_pydatetime().date()
        presence_score = (1 + self.OBS_DECAY_RATE) ** age.days
        return presence_score

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

    # @property
    # def zone_ids(self):
    #     if len(self._zone_ids) < 1:
    #         self._zone_ids = (
    #             self.zones.aggregate_histogram(self.ZONES_LABEL).keys().getInfo()
    #         )
    #     return self._zone_ids

    @property
    def df_adhoc(self):
        if self._df_adhoc is None:
            query = (
                f"SELECT {self.UNIQUE_ID_LABEL}, "
                f"{self.POINT_LOC_LABEL}, "
                f"{self.GRIDCELL_LOC_LABEL}, "
                f"EndObservationDate AS {self.DATE_LABEL} "
                f"FROM dbo.vw_CI_AdHocObservation "
                f"WHERE DATEDIFF(YEAR, EndObservationDate, '{self.taskdate}') <= {self.inputs['obs_adhoc']['maxage']} "
                f"AND EndObservationDate <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_adhoc = self._get_df(query)

        return self._df_adhoc

    @property
    def df_cameratrap(self):
        if self._df_ct is None:
            query = (
                f"SELECT {self.UNIQUE_ID_LABEL}, "
                f"CameraTrapDeploymentID, "
                f"{self.POINT_LOC_LABEL}, "
                f"{self.GRIDCELL_LOC_LABEL}, "
                f"PickupDatetime AS {self.DATE_LABEL} "
                f"FROM dbo.vw_CI_CameraTrapDeployment "
                f"WHERE DATEDIFF(YEAR, PickupDatetime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
                f"AND PickupDatetime <= Cast('{self.taskdate}' AS datetime) "
            )
            df_ct_dep = self._get_df(query)
            df_ct_dep.set_index("CameraTrapDeploymentID", inplace=True)

            query = (
                f"SELECT * FROM dbo.vw_CI_CameraTrapObservation "
                f"WHERE DATEDIFF(YEAR, ObservationDate, '{self.taskdate}') <= "
                f"{self.inputs['obs_ct']['maxage']} "
                f"AND ObservationDate <= Cast('{self.taskdate}' AS date) "
            )
            df_ct_obs = self._get_df(query)

            df_ct_obs.set_index("CameraTrapDeploymentID", inplace=True)
            df_ct_obs["detections"] = (
                df_ct_obs["AdultMaleCount"]
                + df_ct_obs["AdultFemaleCount"]
                + df_ct_obs["AdultSexUnknownCount"]
                + df_ct_obs["SubAdultCount"]
                + df_ct_obs["YoungCount"]
            )
            df_ct_obs["detections"].fillna(0, inplace=True)

            # number of days with > 0 detections per camera
            df_ct_obsdays = (
                df_ct_obs[df_ct_obs["detections"] > 0]
                .groupby(["CameraTrapDeploymentID"])
                .count()
            )
            _df_ct = pd.merge(
                left=df_ct_dep,
                right=df_ct_obsdays,
                left_index=True,
                right_index=True,
                how="left",
            )
            _df_ct["detections"].fillna(0, inplace=True)
            _df_ct.rename(columns={"UniqueID_x": self.UNIQUE_ID_LABEL}, inplace=True)
            self._df_ct = _df_ct.loc[_df_ct["detections"] > 0]

        return self._df_ct

    @property
    def df_signsurvey(self):
        if self._df_ss is None:
            query = (
                f"SELECT {self.UNIQUE_ID_LABEL}, "
                f"{self.POINT_LOC_LABEL}, "
                f"{self.GRIDCELL_LOC_LABEL}, "
                f"StartDate AS {self.DATE_LABEL}, "
                f"detections "
                f"FROM dbo.vw_CI_SignSurveyObservation "
                f"WHERE DATEDIFF(YEAR, StartDate, '{self.taskdate}') <= {self.inputs['obs_ss']['maxage']} "
                f"AND StartDate <= Cast('{self.taskdate}' AS datetime) "
            )
            _df_ss = self._get_df(query)
            self._df_ss = _df_ss.loc[_df_ss["detections"] > 0]

        return self._df_ss

    @property
    def fc_observations(self):
        if self._fc_observations is None:
            obs_dfs = []
            for df in [self.df_adhoc, self.df_cameratrap, self.df_signsurvey]:
                # df with point geom if we have one, polygon otherwise, or drop if neither
                obs_df = df[
                    [
                        self.UNIQUE_ID_LABEL,
                        self.POINT_LOC_LABEL,
                        self.GRIDCELL_LOC_LABEL,
                        self.DATE_LABEL,
                    ]
                ]
                obs_df = obs_df.dropna(
                    subset=[
                        self.POINT_LOC_LABEL,
                        self.GRIDCELL_LOC_LABEL,
                        self.DATE_LABEL,
                    ],
                    how="all",
                )
                obs_df = obs_df.assign(geom=obs_df[self.POINT_LOC_LABEL])
                obs_df.loc[obs_df["geom"].isna(), "geom"] = obs_df[
                    self.GRIDCELL_LOC_LABEL
                ]
                obs_df["presence_score"] = obs_df.apply(
                    lambda row: self._obs_score(row), axis=1
                )
                obs_df = obs_df[[self.UNIQUE_ID_LABEL, "geom", "presence_score"]]
                obs_df.set_index(self.UNIQUE_ID_LABEL, inplace=True)
                # print(obs_df)
                obs_dfs.append(obs_df)

            merged_obs_df = pd.concat(obs_dfs)
            self._fc_observations = self.df2fc(merged_obs_df)

        return self._fc_observations

    def presence_score(self, sclpoly):
        obs_in_sclpoly = ee.Join.simple().apply(
            self.fc_observations, sclpoly, self.intersects
        )
        score_true = obs_in_sclpoly.aggregate_sum("presence_score")
        score = ee.Algorithms.If(obs_in_sclpoly.size().gte(1), score_true, 0)
        return sclpoly.set("presence_score", score)

    def calc(self):
        # TODO: Should the different dfs be treated differently?
        # TODO: evaluate score assigned by time-weighting function
        # print(self.scl)

        scl_scored = self.scl.map(self.presence_score)
        # TODO: Still need to account for effort
        scl_species = scl_scored.filter(self.scl_poly_filters["scl_species"])
        scl_survey = scl_scored.filter(self.scl_poly_filters["scl_survey"])
        scl_restoration = scl_scored.filter(self.scl_poly_filters["scl_restoration"])
        scl_fragment = scl_scored.filter(self.scl_poly_filters["scl_fragment"])

        self.poly_export(scl_species, "scl_species")
        self.poly_export(scl_survey, "scl_survey")
        self.poly_export(scl_restoration, "scl_restoration")
        self.poly_export(scl_fragment, "scl_fragment")

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
    parser.add_argument("--scenario")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing outputs instead of incrementing",
    )
    options = parser.parse_args()
    sclclassification_task = SCLClassification(**vars(options))
    sclclassification_task.run()
