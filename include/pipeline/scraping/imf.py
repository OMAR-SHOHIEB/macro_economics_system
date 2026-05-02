"""This file called imf but i made scaraping form world bank.I made it because i will make some changes in the future.
I need the pipeline in the good standart to improve it in the future in "scalability" """

import requests
import pandas as pd
from countries import retrun_countries
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from include.pipeline.storage.save_csv import imf_save

countries = retrun_countries()

def get_exchange_rate_panel(countries):
    all_data = []
    indicator = "PA.NUS.FCRF"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for country_name, country_code in countries.items():
        url = (
            f"https://api.worldbank.org/v2/country/"
            f"{country_code}/indicator/{indicator}"
            f"?format=json&per_page=500"
        )

        try:
            r = requests.get(url, headers=headers, timeout=20)

            if r.status_code != 200:
                print(f"{country_name}: HTTP {r.status_code}")
                continue

            if not r.text.strip():
                print(f"{country_name}: empty response")
                continue

            response = r.json()

            if len(response) < 2:
                print(f"{country_name}: no data")
                continue

            for row in response[1]:
                if row["value"] is None:
                    continue

                year = int(row["date"])

                if 1980 <= year <= 2024:
                    all_data.append({
                        "country": country_name,
                        "country_code": country_code,
                        "year": year,
                        "exchange_rate": row["value"]
                    })

            print(f"Done -> {country_name}")

        except Exception as e:
            print(f"Error in {country_name}: {e}")

    df = pd.DataFrame(all_data)

    return df.sort_values(["country", "year"]).reset_index(drop=True)

df_exchange = get_exchange_rate_panel(countries)
imf_save(df_exchange)