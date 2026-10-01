# SPDX-FileCopyrightText: 2025-2026 AgroVision <s.maurogiovanni@fz-juelich.de>
#
# SPDX-License-Identifier: MIT

from enum import StrEnum, auto

import polars as pl

lucas_gisco_url = "https://gisco-services.ec.europa.eu/lucas/photos"


LUCAS_METADATA_SCHEMA = {
    "POINT_ID": pl.Utf8,
    "Country": pl.Utf8,
    "Year": pl.Int16,
    "LAT_USED": pl.Float64,
    "LON_USED": pl.Float64,
    "LC1": pl.Utf8,
    "LU1": pl.Utf8,
    "LC2": pl.Utf8,
    "LU2": pl.Utf8,
    "QUALITY_CLASS": pl.Categorical,
}


class LucasSuffix(StrEnum):
    NORTH = auto()
    SOUTH = auto()
    EAST = auto()
    WEST = auto()
    POINT = auto()
    CLOSEUP = auto()

    @property
    def value_2018(self):
        return self.value[0].upper()

    @property
    def value_2022(self):
        suffix = "LC1" if self.value == "closeup" else ""
        name = "crop" if self.value == "closeup" else self.value

        return f"LCLU_{name.title()}{suffix}"


def extend_metadata_frame(
    input_frame: pl.DataFrame, year: int, suffix: LucasSuffix
) -> pl.DataFrame:
    year_prefix = "2022" if year == 2022 else ""
    direction_suffix = suffix.value_2022 if year == 2022 else suffix.value_2018

    url_expr = pl.format(
        "{}/{}/{}/{}/{}/{}{}{}.jpg",
        pl.lit(lucas_gisco_url),
        pl.lit(year),
        pl.col("Country"),
        pl.col("POINT_ID").str.slice(0, 3),
        pl.col("POINT_ID").str.slice(3, 3),
        pl.lit(year_prefix),
        pl.col("POINT_ID"),
        pl.lit(direction_suffix),
    )

    filename_expr = pl.format(
        "{}-{}-{}-{}.jpg",
        pl.lit(year),
        pl.col("POINT_ID"),
        pl.col("Country").str.to_lowercase(),
        pl.lit(suffix.value),
    )

    return input_frame.with_columns(
        url_expr.alias("url"), filename_expr.alias("filename")
    )
