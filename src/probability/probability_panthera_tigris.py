import pathlib
import numpy as np
import pandas as pd
import pyjags as pj
from functools import reduce
from includes.constants import *
import arviz as az


ITERATIONS = 15000
BURNIN = 5000
MODEL_PARAMETERS = [PHI0, "b_area", "b_proportion_protected", "beta0", "beta", "beta2"]


def create_df(cam, sign, df_adhoc, df_poly):
    df_poly.sort_values(by=[SCLPOLY_ID], inplace=True)
    # Get the number of rows for a given poly_id where
    # For a given polygon cam detections > 0
    cam_detection = (
        cam.query(f"{CT_DAYS_DETECTED} > 0").groupby(SCLPOLY_ID).size().to_frame("cam")
    )
    sign_detection = (
        sign.query(f"{SS_SEGMENTS_DETECTED} > 0")
        .groupby(SCLPOLY_ID)
        .size()
        .to_frame("sign")
    )
    # For a given poly and cam where days > 0
    days = (
        cam.query(f"{CT_DAYS_OPERATING} > 0")
        .groupby(SCLPOLY_ID)
        .size()
        .to_frame("days")
    )

    adhoc_subset = (
        df_adhoc.query(f"{DENSITY} != [0.0]")
        .groupby(SCLPOLY_ID)
        .size()
        .to_frame("adhoc")
    )
    # For a given poly and sign Number of Replicates surveyed > 0
    sign_replicates = (
        sign.query(f"{SS_SEGMENTS_SURVEYED} > 0")
        .groupby(SCLPOLY_ID)
        .size()
        .to_frame("replicates")
    )

    dfs_to_join = [cam_detection, sign_detection, sign_replicates, days, adhoc_subset]
    df_detections = reduce(
        lambda left, right: pd.merge(left, right, on=SCLPOLY_ID, how="outer"),
        dfs_to_join,
    )
    df_poly_detections = pd.merge(df_poly, df_detections, on=SCLPOLY_ID, how="left")
    df_poly_detections = df_poly_detections.fillna(0)

    # Assign known_occ 1/0 based on sum of SS, CT, and ad hoc weighted by a
    # threshold representing how many observations should count as definitive presence
    known_occ = (
        (df_poly_detections["adhoc"] * 1 / ADHOC_COUNT_THRESHOLD)
        + df_poly_detections["sign"]
        + df_poly_detections["cam"]
    )
    df_poly_detections["known_occ"] = np.where(known_occ >= 1, 1, 0)
    # If the sum of known_occ, replicates and days is > 0, assign 1 else 0
    surveyed = (
        df_poly_detections["known_occ"]
        + df_poly_detections["replicates"]
        + df_poly_detections["days"]
    )
    df_poly_detections["surveyed"] = np.where(surveyed > 0, 1, 0)
    only_ah = (df_poly_detections["known_occ"] == 1) & (
        (df_poly_detections["sign"] + df_poly_detections["cam"]) == 0
    )
    df_poly_detections["only_ah"] = np.where(only_ah, 1, 0)

    return df_poly_detections


def get_unique_values(df_polys, sign, cam):
    unique_values_dict = dict()
    unique_values_dict["num_poly"] = len(df_polys)

    countries = np.unique(df_polys[COUNTRY])
    unique_values_dict["ordered_unique_countries"] = countries
    # Index position of unique country for each country value
    unique_values_dict[COUNTRY] = [
        list(countries).index(i) + 1 for i in df_polys[COUNTRY]
    ]
    unique_values_dict["num_country"] = len(countries)

    biomes = np.unique(df_polys[BIOME])
    unique_values_dict["ordered_unique_biomes"] = biomes
    # Index position of unique biomes for each biome value
    unique_values_dict[BIOME] = [list(biomes).index(i) + 1 for i in df_polys[BIOME]]
    unique_values_dict["num_biome"] = len(biomes)

    # get unique values of mastergridcell from sign and from cam combined
    sign_master_grid = pd.Series(pd.unique(sign[MASTER_CELL])).sort_values()
    cam_master_grid = pd.Series(pd.unique(cam[MASTER_CELL])).sort_values()
    uni_det = pd.concat([sign_master_grid, cam_master_grid])
    unique_values_dict["uni_det"] = pd.unique(uni_det)
    unique_values_dict["num_surv_grid"] = len(pd.unique(uni_det))

    return unique_values_dict


def assign_grid_values(mastergridcell, sign, cam):
    # Get the number of rows for a given poly_id where
    sign_grid = sign.loc[(sign[MASTER_CELL] == mastergridcell)]
    cam_grid = cam.loc[(cam[MASTER_CELL] == mastergridcell)]
    num_sign_grid = len(sign_grid)
    master_grid_cell = mastergridcell

    if num_sign_grid == 0:
        # take the index i in tpoly_id
        # assign this the first value of cam_grid's poly_id
        # use iloc[0] here to get first value based on indexing
        tpoly_id = cam_grid[SCLPOLY_ID].iloc[0]
        nsdet = 0
        nstrials = 0
        ncdet = cam_grid[CT_DAYS_DETECTED].sum()
        nctrials = cam_grid[CT_DAYS_OPERATING].sum()
        if ncdet > 0:
            known_use = 1
        else:
            known_use = 0

    else:
        # assign the first poly id from the temp table
        tpoly_id = sign_grid[SCLPOLY_ID].iloc[0]
        nsdet = sign_grid[SS_SEGMENTS_DETECTED].sum()
        nstrials = sign_grid[SS_SEGMENTS_SURVEYED].sum()
        ncdet = cam_grid[CT_DAYS_DETECTED].sum()
        nctrials = cam_grid[CT_DAYS_OPERATING].sum()
        if (ncdet + nsdet) > 0:
            known_use = 1
        else:
            known_use = 0
    ns = [master_grid_cell, tpoly_id, nsdet, nstrials, ncdet, nctrials, known_use]

    return ns


def create_df_grid(uni_det, sign, cam):
    dd = [assign_grid_values(mastergridcell=x, sign=sign, cam=cam) for x in uni_det]
    df = pd.DataFrame(
        dd,
        columns=[
            MASTER_CELL,
            "tpoly_id",
            "nsdet",
            "nstrials",
            "ncdet",
            "nctrials",
            "known_use",
        ],
    )
    return df


def format_data(df_polys, df_adhoc, df_cameratrap, df_signsurvey):
    # Filter sign/cam rows that don't fall within polys
    sign = df_signsurvey.loc[df_signsurvey[SCLPOLY_ID] != NODATA]
    cam = df_cameratrap.loc[df_cameratrap[SCLPOLY_ID] != NODATA]
    sign = sign[pd.isna(sign[SCLPOLY_ID]) == False]
    cam = cam[pd.isna(cam[SCLPOLY_ID]) == False]

    # Get the number and unique values for model inputs
    unique_values = get_unique_values(df_polys, sign, cam)

    # Create a dataframe for each poly_id
    df_poly_detections = create_df(cam, sign, df_adhoc, df_polys)
    # replace columns of df_poly_detections with adjusted unique values to account for starting at 1 instead of 0
    df_poly_detections[BIOME] = unique_values[BIOME]
    df_poly_detections[COUNTRY] = unique_values[COUNTRY]

    # Access dict to get the value to pass to create_df_grid for mastergridcells
    uni_det = unique_values["uni_det"]
    gridded_df = create_df_grid(uni_det, sign, cam)

    # Standardized area (0-1)
    area_std = (df_polys[HABITAT_AREA] - np.mean(df_polys[HABITAT_AREA])) / np.std(
        df_polys[HABITAT_AREA]
    )

    # data dictionary for JAGS input, additional checks for trials, positive integers needed
    jags_input_data = {
        "Npoly": unique_values["num_poly"],
        COUNTRY: unique_values[COUNTRY],
        BIOME: unique_values[BIOME],
        "Nsurvgrid": unique_values["num_surv_grid"],
        SCLPOLY_ID: gridded_df["tpoly_id"],
        "Ncountry": unique_values["num_country"],
        "Nbiome": unique_values["num_biome"],
        "nsdet": gridded_df["nsdet"],
        "ncdet": gridded_df["ncdet"],
        "nctrials": round(abs(gridded_df["nctrials"])),
        "nstrials": round(abs(gridded_df["nstrials"])),
        "area_std": area_std,
        "proportion_protected": df_polys[PROPORTION_PROTECTED],
    }

    jags_initial_values = dict(
        phi=df_poly_detections["known_occ"], p_use=gridded_df["known_use"]
    )

    fixed_effects_stats = {
        # columns need to align with output of az.summary()
        "area": [
            np.mean(df_polys[HABITAT_AREA]),
            np.std(df_polys[HABITAT_AREA]),
        ]
        + [None] * 7
        # no need for proportion protected because it's not pre-standardized
    }

    return (
        jags_input_data,
        df_poly_detections,
        jags_initial_values,
        unique_values["ordered_unique_biomes"],
        unique_values["ordered_unique_countries"],
        fixed_effects_stats,
    )


def run_jags_model(jags_data_formatted):
    current_dir = pathlib.Path(__file__).parent.resolve()
    model = pj.Model(
        file=f"{current_dir}/panthera_tigris.jags",
        chains=3,
        data=jags_data_formatted[0],
        init=jags_data_formatted[2],
    )
    # only retain posterior draws from these parameters
    parameters = [PROBABILITY] + MODEL_PARAMETERS
    samples = model.sample(iterations=ITERATIONS, vars=parameters)
    samples_after_burn_in = pj.discard_burn_in_samples(samples, burn_in=BURNIN)

    return samples_after_burn_in


def parameter_diagnostics(jags_output, fixed_effects_stats):
    print("Creating model diagnostics")

    pyjags_data = az.from_pyjags(jags_output)
    pyjags_diagnostics = az.summary(pyjags_data["posterior"][MODEL_PARAMETERS])
    for label, stats in fixed_effects_stats.items():
        pyjags_diagnostics.loc[label] = stats

    return pyjags_diagnostics


def jags_post_process(jags_output, df_poly_detections):
    # phi output, get mean over all iterations for each chain
    phi_mean = []
    for i in range(len(jags_output[PROBABILITY])):
        phi_mean.append(jags_output[PROBABILITY][i].flatten().mean())

    phi_mean_round = [round(100 * i) for i in phi_mean]

    # modify all adhoc-only poly_ids to be 100
    for i in range(len(df_poly_detections["only_ah"])):
        if df_poly_detections["only_ah"][i] == 1:
            phi_mean_round[i] = 100

    # phi0 output, get mean over all iterations for each chain
    phi0_mean = []
    for i in range(len(jags_output[PHI0])):
        phi0_mean.append(jags_output[PHI0][i].flatten().mean())

    phi0_mean_round = [round(100 * i) for i in phi0_mean]

    effort = [0] * len(jags_output[PROBABILITY])
    for i in range(len(jags_output[PROBABILITY])):
        if (
            df_poly_detections["surveyed"][i] == 1
            and df_poly_detections["known_occ"][i] == 0
        ):
            effort[i] = round(100 - 100 * ((phi_mean[i]) / (phi0_mean[i])))

    out = pd.DataFrame(
        {
            SCLPOLY_ID: df_poly_detections[SCLPOLY_ID],
            "biome_index": df_poly_detections[BIOME],
            "country_index": df_poly_detections[COUNTRY],
            "known_occ": df_poly_detections["known_occ"],
            "only_ah": df_poly_detections["only_ah"],
            "surveyed": df_poly_detections["surveyed"],
            PROBABILITY: phi_mean_round,
            "unconditional_prob": phi0_mean_round,
            EFFORT: effort,
        }
    )
    out.set_index(SCLPOLY_ID, inplace=True)

    return out


def assign_probabilities(df_polys, df_adhoc, df_cameratrap, df_signsurvey):
    # returns data dictionary for JAGS input, poly detections dataframe, initial values for JAGS input
    jags_data_formatted = format_data(
        df_polys=df_polys,
        df_adhoc=df_adhoc,
        df_cameratrap=df_cameratrap,
        df_signsurvey=df_signsurvey,
    )

    # runs the JAGS model, adjusting for burn-in period and total iterations (predetermined)
    jags_output = run_jags_model(jags_data_formatted=jags_data_formatted)
    diagnostics = parameter_diagnostics(jags_output, jags_data_formatted[5])

    # processes posterior output to desired output format with probabilities and effort as integers
    # output dataframe includes: poly_id, biome, country, known_occ, only_ah, surveyed, phi, effort
    jags_processed = jags_post_process(
        jags_output=jags_output, df_poly_detections=jags_data_formatted[1]
    )

    model_metadata = {
        "diagnostics": diagnostics,
        "ordered_unique_biomes": pd.DataFrame(
            jags_data_formatted[3], columns=["biome_code"]
        ),
        "ordered_unique_countries": pd.DataFrame(
            jags_data_formatted[4], columns=["country_code"]
        ),
    }

    return jags_processed, model_metadata
