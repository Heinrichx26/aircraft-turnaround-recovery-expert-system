from pathlib import Path

import pandas as pd

from operational_records import BASE_USECOLS, MODEL_CATEGORICAL, MODEL_NUMERIC


def load_turnarounds(path: Path, horizon: int) -> pd.DataFrame:
    needed = sorted(
        set(
            BASE_USECOLS
            + MODEL_NUMERIC
            + MODEL_CATEGORICAL
            + [
                f"recover_h{horizon}",
                f"fail_h{horizon}",
                f"endpoint_obs_h{horizon}",
            ]
        )
    )
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in needed if column in set(header.columns)]
    turnarounds = pd.read_csv(path, usecols=usecols, low_memory=False)
    turnarounds["sched_dep_dt"] = pd.to_datetime(turnarounds["sched_dep_dt"], errors="coerce")
    for column in [
        *MODEL_NUMERIC,
        f"recover_h{horizon}",
        f"fail_h{horizon}",
        f"endpoint_obs_h{horizon}",
        "is_cancelled",
        "is_diverted",
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "late_aircraft_delay",
    ]:
        if column in turnarounds.columns:
            turnarounds[column] = pd.to_numeric(turnarounds[column], errors="coerce")
    for column in MODEL_CATEGORICAL + ["episode_id", "tail"]:
        if column in turnarounds.columns:
            turnarounds[column] = turnarounds[column].fillna("UNK").astype(str)
    return turnarounds
