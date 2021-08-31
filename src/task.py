import argparse
import ee
import os
import re
import subprocess
import numpy as np
import pandas as pd
import pyodbc
# import time
# import uuid
# import json
from datetime import datetime, timezone
from typing import Optional, Union
# from geomet import wkt
from google.cloud.storage import Client
from google.cloud.exceptions import NotFound
from pathlib import Path
from task_base import SCLTask


# noinspection PyTypeChecker,PyUnresolvedReferences
class SCLClassification(SCLTask):
    google_creds_path = "/.google_creds"
    inputs = {
        "obs_adhoc": {"maxage": 50},
        "obs_ss": {"maxage": 50},
        "obs_ct": {"maxage": 50},
        "zones": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "scl_zones",
            "static": True,
        },
        "scl": {
            "ee_type": SCLTask.FEATURECOLLECTION,
            "ee_path": "scl_polys",
            "maxage": 1 / 365,
        },
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

        self._df_adhoc = self._df_ct = self._df_ss = self._df_covars = None
        self._zone_ids = []
        self.zone = None
        self.fc_csvs = []
        self.zones = ee.FeatureCollection(self.inputs["zones"]["ee_path"])
        self.scl = ee.FeatureCollection(self.inputs["scl"]["ee_path"])

    # TODO: move to task_base
    def scl_zones(self):
        return f"projects/{self.ee_project}/{self.species}/zones"

    def scl_polys(self):
        return f"{self.ee_rootdir}/pothab/scl_polys"

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
                f"SELECT * FROM dbo.vw_CI_AdHocObservation "
                f"WHERE DATEDIFF(YEAR, EndObservationDate, '{self.taskdate}') <= {self.inputs['obs_adhoc']['maxage']} "
                f"AND EndObservationDate <= Cast('{self.taskdate}' AS datetime) "
            )
            self._df_adhoc = self._get_df(query)

        return self._df_adhoc

    # @property
    # def df_cameratrap(self):
    #     if self._df_ct is None:
    #         _csvpath = "cameratrap.csv"
    #         if self.use_cache and Path(_csvpath).is_file():
    #             self._df_ct = pd.read_csv(
    #                 _csvpath, encoding="utf-8", index_col="CameraTrapDeploymentID"
    #             )
    #         else:
    #             query = (
    #                 f"SELECT * FROM dbo.vw_CI_CameraTrapDeployment "
    #                 f"WHERE DATEDIFF(YEAR, PickupDatetime, '{self.taskdate}') <= {self.inputs['obs_ct']['maxage']} "
    #                 f"AND PickupDatetime <= Cast('{self.taskdate}' AS datetime) "
    #             )
    #             df_ct_dep = self._get_df(query)
    #             print("zonify camera trap deployments")
    #             df_ct_dep = self.zonify(df_ct_dep)
    #             df_ct_dep.set_index("CameraTrapDeploymentID", inplace=True)
    #
    #             query = (
    #                 f"SELECT * FROM dbo.vw_CI_CameraTrapObservation "
    #                 f"WHERE DATEDIFF(YEAR, ObservationDate, '{self.taskdate}') <= "
    #                 f"{self.inputs['obs_ct']['maxage']} "
    #                 f"AND ObservationDate <= Cast('{self.taskdate}' AS date) "
    #             )
    #             df_ct_obs = self._get_df(query)
    #
    #             df_ct_obs.set_index("CameraTrapDeploymentID", inplace=True)
    #             df_ct_obs["detections"] = (
    #                 df_ct_obs["AdultMaleCount"]
    #                 + df_ct_obs["AdultFemaleCount"]
    #                 + df_ct_obs["AdultSexUnknownCount"]
    #                 + df_ct_obs["SubAdultCount"]
    #                 + df_ct_obs["YoungCount"]
    #             )
    #             df_ct_obs["detections"].fillna(0, inplace=True)
    #
    #             # number of days with > 0 detections per camera
    #             df_ct_obsdays = (
    #                 df_ct_obs[df_ct_obs["detections"] > 0]
    #                 .groupby(["CameraTrapDeploymentID"])
    #                 .count()
    #             )
    #             self._df_ct = pd.merge(
    #                 left=df_ct_dep,
    #                 right=df_ct_obsdays,
    #                 left_index=True,
    #                 right_index=True,
    #                 how="left",
    #             )
    #             self._df_ct["detections"].fillna(0, inplace=True)
    #
    #         if self.use_cache and not self._df_ct.empty:
    #             self._df_ct.to_csv(_csvpath)
    #
    #     return self._df_ct[self._df_ct[self.MASTER_GRID_LABEL].astype(str) == self.zone]
    #
    # @property
    # def df_signsurvey(self):
    #     if self._df_ss is None:
    #         _csvpath = "signsurvey.csv"
    #         if self.use_cache and Path(_csvpath).is_file():
    #             self._df_ss = pd.read_csv(
    #                 _csvpath, encoding="utf-8", index_col=self.MASTER_CELL_LABEL
    #             )
    #         else:
    #             query = (
    #                 f"SELECT * FROM dbo.vw_CI_SignSurveyObservation "
    #                 f"WHERE DATEDIFF(YEAR, StartDate, '{self.taskdate}') <= {self.inputs['obs_ss']['maxage']} "
    #                 f"AND StartDate <= Cast('{self.taskdate}' AS datetime) "
    #             )
    #             self._df_ss = self._get_df(query)
    #             print("zonify sign survey")
    #             self._df_ss = self.zonify(self._df_ss)
    #             self._df_ss.set_index(self.MASTER_CELL_LABEL, inplace=True)
    #
    #         if self.use_cache and not self._df_ss.empty:
    #             self._df_ss.to_csv(_csvpath)
    #
    #     return self._df_ss[self._df_ss[self.MASTER_GRID_LABEL].astype(str) == self.zone]

    def calc(self):
        print(self.df_adhoc)
        print(self.scl)
        # establish time-weighting function
        # create fc for combined obs df (adhoc, signsurvey, cameratrap)
        # - with score assigned by time-weighting function
        # - filtered for positive observations if necessary (no no_detection points)
        # map function: for every scl polygon:
        # - obs df.filterBounds(scl polygon)
        # - if at least one observation in polygon has score > threshold, assign polygon "probability" (or rename) = 1
        # use eff_pot_hab classification filters to classify
        # - >>> Some way to capture effort? 

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
