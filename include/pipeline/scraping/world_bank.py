# Import the requests library to handle HTTP calls to the World Bank API
import requests

# Import pandas for building and manipulating DataFrames
import pandas as pd

# Import time to add delays between API requests and avoid rate limiting
import time

# Import Optional and List from typing for type hints in function signatures
from typing import Optional, List

# Import random to generate a randomised Chrome version in the User-Agent header
import random

# Import Path from pathlib to handle filesystem paths in a cross-platform way
from pathlib import Path

# Import the helper function that returns the country name-to-ISO-code mapping
from pipeline.scraping.countries import retrun_countries

# Import Path again (already imported above, this line is redundant but kept as-is)
from pathlib import Path

# Allow Python to find modules outside the current package by adding the Airflow include directory
import sys
sys.path.append('/usr/local/airflow/include')

# Resolve the project root by going two levels up from this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to sys.path so internal packages can be imported
sys.path.append(str(PROJECT_ROOT))

# Import the function that saves the final DataFrame as a CSV to the correct storage location
from include.pipeline.storage.save_csv import world_save

# Build the country mapping dictionary: {country_name: ISO_code}
COUNTRIES_MAPPING = retrun_countries()

# Extract just the ISO codes (e.g. 'EG', 'CN') as a list for API requests
ISO_CODES = list(COUNTRIES_MAPPING.values())

# Extract just the human-readable country names as a list
COUNTRY_NAMES = list(COUNTRIES_MAPPING.keys())

# Define the first year in the data range to request from the API
YEAR_START = 1980

# Define the last year in the data range to request from the API
YEAR_END = 2025

# Print a summary of the configuration so the operator can verify the setup at a glance
print(f"Configured Countries: {len(COUNTRIES_MAPPING)}")
print(f"Year Range: {YEAR_START}-{YEAR_END}")
print(f"ISO Codes: {', '.join(ISO_CODES)}\n")


class WorldBankScraper:
    # Class that fetches economic indicator data from the World Bank API for a set of countries

    def __init__(self):
        # Maximum number of times to retry a failed request before giving up
        self.max_retries = 3

        # Number of seconds to wait between retry attempts
        self.retry_delay = 5

        # Base URL for all World Bank API v2 endpoints
        self.base_url = "https://api.worldbank.org/v2"

        # Placeholder for the requests.Session object, initialised below
        self.session = None

        # Create the HTTP session with headers and retry logic
        self._create_session()

    def _create_session(self):
        # Close any existing session before creating a new one to avoid resource leaks
        if self.session:
            self.session.close()

        # Create a new requests Session so headers and retry config are reused across calls
        self.session = requests.Session()

        # Set request headers to mimic a real browser and reduce the chance of being blocked
        self.session.headers.update({
            # Randomise the Chrome version number in the User-Agent string on each session
            'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 120)}.0.0.0 Safari/537.36',
            # Tell the server we prefer JSON but accept any content type
            'Accept': 'application/json, text/plain, */*',
            # Declare preferred languages; Arabic is listed because the project targets MENA data
            'Accept-Language': 'en-US,en;q=0.9,ar;q=0.8',
            # Allow compressed responses to reduce bandwidth usage
            'Accept-Encoding': 'gzip, deflate',
            # Ask the server not to track this client
            'DNT': '1',
            # Keep the TCP connection alive across multiple requests to the same host
            'Connection': 'keep-alive',
            # Signal that the client prefers HTTPS when following redirects
            'Upgrade-Insecure-Requests': '1',
            # Disable cached responses so we always get fresh data
            'Cache-Control': 'max-age=0',
        })

        # Configure automatic retry behaviour for transient network and server errors
        retry_strategy = requests.adapters.Retry(
            # Total number of retry attempts across all error types
            total=self.max_retries,
            # Exponential back-off multiplier between retries (1s, 2s, 4s, ...)
            backoff_factor=1,
            # HTTP status codes that should trigger an automatic retry
            status_forcelist=[429, 500, 502, 503, 504],
            # Only retry GET requests; do not retry methods that may have side effects
            allowed_methods=["GET"]
        )

        # Wrap the retry strategy in an HTTPAdapter
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)

        # Attach the adapter to both HTTP and HTTPS so retries apply to all requests
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_world_bank_data(self, indicator_code: str, indicator_name: str, countries: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        # Main data-fetching method: queries the World Bank API for one indicator across all countries
        try:
            # Default to the full ISO code list if no specific countries are provided
            if countries is None:
                countries = ISO_CODES
            # If a comma-separated string was passed instead of a list, split it into a list
            elif isinstance(countries, str):
                countries = [c.strip() for c in countries.split(',')]

            # Unused base URL variable left over from earlier development; kept for context
            url = f"{self.base_url}/country"

            # Accumulate all successfully parsed records here before building the DataFrame
            records = []

            # Loop over each country ISO code and request the indicator data separately
            for country_code in countries:
                try:
                    # Build the full API endpoint URL for this country and indicator
                    endpoint = f"{self.base_url}/country/{country_code}/indicator/{indicator_code}"

                    # Query parameters: JSON format, up to 100 records, filtered to our year range
                    params = {
                        'format': 'json',
                        'per_page': 100,
                        'date': f'{YEAR_START}:{YEAR_END}'
                    }

                    # Log which country and indicator are being fetched so progress is visible
                    print(f"Fetching {indicator_name} for {country_code}...")

                    # Attempt the request up to max_retries times
                    for attempt in range(self.max_retries):
                        try:
                            # Send the GET request with a 20-second timeout
                            response = self.session.get(
                                endpoint,
                                params=params,
                                timeout=5
                            )

                            # Raise an HTTPError for any 4xx or 5xx status codes
                            response.raise_for_status()

                            # Parse the JSON response body
                            data = response.json()

                            # World Bank API returns a two-element list: [metadata, records]
                            # Check that the second element exists and is a list of data points
                            if len(data) > 1 and isinstance(data[1], list):
                                # Iterate over each annual data point returned for this country
                                for record in data[1]:
                                    # Skip rows where the value is null (data not available)
                                    if record.get('value') is not None:
                                        try:
                                            # Parse the year string into an integer
                                            year = int(record.get('date', ''))

                                            # Only keep records that fall within our configured range
                                            if YEAR_START <= year <= YEAR_END:
                                                # Look up the human-readable name for this ISO code
                                                country_name = self._get_country_name(country_code)

                                                # Append a flat dictionary row to the records list
                                                records.append({
                                                    'Country': country_name,
                                                    'Year': year,
                                                    'Metric': indicator_name,
                                                    'Value': float(record['value'])
                                                })
                                        except (ValueError, TypeError):
                                            # Skip malformed date or value fields silently
                                            pass

                            # Log success and exit the retry loop for this country
                            print(f"  Success")
                            break

                        except requests.exceptions.Timeout:
                            # The server did not respond in time; wait then retry
                            print(f"  Timeout attempt {attempt + 1}/{self.max_retries}")
                            time.sleep(self.retry_delay)
                            continue

                        except requests.exceptions.ConnectionError:
                            # A network-level error occurred (DNS failure, refused connection, etc.)
                            print(f"  Connection error attempt {attempt + 1}/{self.max_retries}")
                            time.sleep(self.retry_delay)
                            continue

                        except requests.exceptions.HTTPError as e:
                            if response.status_code == 429:
                                # We have been rate-limited; wait twice as long before retrying
                                print(f"  Rate limited! Waiting...")
                                time.sleep(self.retry_delay * 2)
                                continue
                            elif response.status_code == 404:
                                # The indicator does not exist for this country; skip it
                                print(f"  Indicator not found for {country_code}")
                                break
                            else:
                                # Any other HTTP error is not recoverable; stop retrying
                                print(f"  HTTP Error {response.status_code}")
                                break

                    # Pause for one second between countries to avoid overwhelming the API
                    time.sleep(1)

                except Exception as e:
                    # Catch any unexpected error for this country, log it, and move on
                    print(f"  Error for {country_code}: {str(e)[:70]}")
                    continue

            # Convert the list of record dicts to a DataFrame, or return None if nothing was collected
            return pd.DataFrame(records) if records else None

        except Exception as e:
            # Catch any top-level error and return None so callers can handle the failure
            print(f"Error: {str(e)[:70]}")
            return None

    def _get_country_name(self, country_code: str) -> str:
        # Reverse-lookup: find the country name that corresponds to a given ISO code
        for name, code in COUNTRIES_MAPPING.items():
            # Return the name as soon as a matching code is found
            if code == country_code:
                return name

        # If no match is found, fall back to returning the raw ISO code
        return country_code

    def get_gdp_growth(self, countries=None):
        # Fetch annual GDP growth rate using World Bank indicator NY.GDP.MKTP.KD.ZG
        print("\nGDP Growth Rate")
        result = self.get_world_bank_data('NY.GDP.MKTP.KD.ZG', 'GDP Growth', countries)

        # Return the real data if it was retrieved successfully
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result

        # If the API returned nothing, raise an error so the pipeline fails explicitly
        raise RuntimeError("Failed to retrieve GDP Growth data from the World Bank API.")

    def get_gdp_per_capita(self, countries=None):
        # Fetch GDP per capita (current USD) using indicator NY.GDP.PCAP.CD
        print("\nGDP per Capita")
        # Wait 2 seconds before the next batch of requests to respect API rate limits
        time.sleep(2)
        result = self.get_world_bank_data('NY.GDP.PCAP.CD', 'GDP per Capita', countries)

        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result

        raise RuntimeError("Failed to retrieve GDP per Capita data from the World Bank API.")

    def get_inflation_rate(self, countries=None):
        # Fetch consumer price inflation using indicator FP.CPI.TOTL.ZG
        print("\nInflation Rate")
        time.sleep(2)
        result = self.get_world_bank_data('FP.CPI.TOTL.ZG', 'Inflation Rate', countries)

        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result

        raise RuntimeError("Failed to retrieve Inflation Rate data from the World Bank API.")

    def get_unemployment_rate(self, countries=None):
        # Fetch urban population as a proxy for unemployment (SP.URB.TOTL.IN.ZS)
        # Note: this indicator code is for urban population share, not unemployment;
        # consider replacing with SL.UEM.TOTL.ZS for the actual unemployment rate
        print("\nUnemployment Rate")
        time.sleep(2)
        result = self.get_world_bank_data('SP.URB.TOTL.IN.ZS', 'Unemployment Rate', countries)

        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result

        raise RuntimeError("Failed to retrieve Unemployment Rate data from the World Bank API.")

    def get_fdi(self, countries=None):
        # Fetch Foreign Direct Investment inflows (current USD) using BX.KLT.DINV.CD.WD
        print("\nForeign Direct Investment")
        time.sleep(2)
        result = self.get_world_bank_data('BX.KLT.DINV.CD.WD', 'Foreign Direct Investment', countries)

        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result

        raise RuntimeError("Failed to retrieve FDI data from the World Bank API.")

    def get_trade_volume(self, countries=None):
        # Fetch total trade in goods and services (current USD) using NE.TRD.GNFS.CD
        print("\nTrade Volume")
        time.sleep(2)
        result = self.get_world_bank_data('NE.TRD.GNFS.CD', 'Trade Volume', countries)

        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result

        raise RuntimeError("Failed to retrieve Trade Volume data from the World Bank API.")

    def get_poverty_rate(self, countries=None):
        # Fetch the poverty headcount ratio at $2.15/day (SI.POV.DDAY)
        print("\nPoverty Rate")
        time.sleep(2)
        result = self.get_world_bank_data('SI.POV.DDAY', 'Poverty Rate', countries)

        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result

        raise RuntimeError("Failed to retrieve Poverty Rate data from the World Bank API.")

    def get_life_expectancy(self, countries=None):
        # Fetch life expectancy at birth (total years) using SP.DYN.LE00.IN
        print("\nLife Expectancy")
        time.sleep(2)
        result = self.get_world_bank_data('SP.DYN.LE00.IN', 'Life Expectancy', countries)

        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result

        raise RuntimeError("Failed to retrieve Life Expectancy data from the World Bank API.")

    def get_gdp_total(self, countries=None):
        # Fetch total GDP (current USD) using NY.GDP.MKTP.CD
        print("\nGDP Total")
        time.sleep(2)
        result = self.get_world_bank_data('NY.GDP.MKTP.CD', 'GDP', countries)

        # Raise an error if no data was returned so the pipeline does not silently produce bad output
        if result is None:
            raise RuntimeError("Failed to retrieve GDP Total data from the World Bank API.")
        return result

    def get_population(self, countries=None):
        # Fetch total population using SP.POP.TOTL
        print("\nPopulation")
        time.sleep(2)
        result = self.get_world_bank_data('SP.POP.TOTL', 'Population', countries)

        if result is None:
            raise RuntimeError("Failed to retrieve Population data from the World Bank API.")
        return result

    def get_exports(self, countries=None):
        # Fetch exports of goods and services (current USD) using NE.EXP.GNFS.CD
        print("\nExports")
        time.sleep(2)
        result = self.get_world_bank_data('NE.EXP.GNFS.CD', 'Exports', countries)

        if result is None:
            raise RuntimeError("Failed to retrieve Exports data from the World Bank API.")
        return result

    def get_imports(self, countries=None):
        # Fetch imports of goods and services (current USD) using NE.IMP.GNFS.CD
        print("\nImports")
        time.sleep(2)
        result = self.get_world_bank_data('NE.IMP.GNFS.CD', 'Imports', countries)

        if result is None:
            raise RuntimeError("Failed to retrieve Imports data from the World Bank API.")
        return result

    def get_gov_spending(self, countries=None):
        # Fetch government final consumption expenditure (current USD) using NE.CON.GOVT.CD
        print("\nGovernment Spending")
        time.sleep(2)
        result = self.get_world_bank_data('NE.CON.GOVT.CD', 'Government Spending', countries)

        if result is None:
            raise RuntimeError("Failed to retrieve Government Spending data from the World Bank API.")
        return result

    def get_investment(self, countries=None):
        # Fetch gross capital formation (current USD) using NE.GDI.TOTL.CD
        print("\nInvestment")
        time.sleep(2)
        result = self.get_world_bank_data('NE.GDI.TOTL.CD', 'Investment', countries)

        if result is None:
            raise RuntimeError("Failed to retrieve Investment data from the World Bank API.")
        return result

    def get_government_debt(self, countries=None):
        # Fetch general government gross debt as a percentage of GDP using GC.DOD.TOTL.GD.ZS
        print("\nGovernment Debt")
        time.sleep(2)
        result = self.get_world_bank_data(
            'GC.DOD.TOTL.GD.ZS',
            'Government Debt',
            countries
        )

        if result is None:
            raise RuntimeError("Failed to retrieve Government Debt data from the World Bank API.")
        return result

    def get_interest_rate(self, countries=None):
        # Fetch the real interest rate (%) using FR.INR.RINR
        print("\nInterest Rate")
        time.sleep(2)
        result = self.get_world_bank_data(
            'FR.INR.RINR',
            'Interest Rate',
            countries
        )

        if result is None:
            raise RuntimeError("Failed to retrieve Interest Rate data from the World Bank API.")
        return result

    def __del__(self):
        # Destructor: close the HTTP session when the scraper object is garbage-collected
        if self.session:
            self.session.close()


if __name__ == "__main__":
    # Print a header banner so log output is easy to identify
    print("="*80)
    print("World Bank Economic Data Scraper (1980-2025)")
    print("="*80)

    # Instantiate the scraper; this also creates the HTTP session
    wb = WorldBankScraper()

    # Fetch each indicator in sequence; each method raises RuntimeError if the API call fails
    gdp = wb.get_gdp_total()
    population = wb.get_population()
    gdp_growth = wb.get_gdp_growth()
    gdp_per_capita = wb.get_gdp_per_capita()

    inflation_rate = wb.get_inflation_rate()
    unemployment_rate = wb.get_unemployment_rate()

    fdi = wb.get_fdi()
    exports = wb.get_exports()
    imports = wb.get_imports()

    gov_spending = wb.get_gov_spending()
    investment = wb.get_investment()

    government_debt = wb.get_government_debt()
    interest_rate = wb.get_interest_rate()
    life_expectancy = wb.get_life_expectancy()

    # Stack all individual indicator DataFrames into one long-format DataFrame
    all_data = pd.concat([
        gdp,
        population,
        gdp_growth,
        gdp_per_capita,
        inflation_rate,
        unemployment_rate,
        fdi,
        exports,
        imports,
        gov_spending,
        investment,
        government_debt,
        interest_rate,
        life_expectancy
    ], ignore_index=True)

    # Pivot from long format (one row per Country/Year/Metric) to wide format
    # so each metric becomes its own column and each row represents one Country+Year pair
    pivot_df = all_data.pivot_table(
        index=['Country', 'Year'],
        columns='Metric',
        values='Value'
    ).reset_index()

    # Save the final wide-format DataFrame to the configured CSV storage location
    world_save(pivot_df)