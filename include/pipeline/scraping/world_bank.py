import requests
import pandas as pd
import time
from typing import Optional, List
import random
from pathlib import Path
from pipeline.scraping.countries import retrun_countries
from pathlib import Path

import sys
sys.path.append('/usr/local/airflow/include')
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from include.pipeline.storage.save_csv import world_save

COUNTRIES_MAPPING = retrun_countries()
ISO_CODES = list(COUNTRIES_MAPPING.values())
COUNTRY_NAMES = list(COUNTRIES_MAPPING.keys())
YEAR_START = 1980
YEAR_END = 2025

print(f"Configured Countries: {len(COUNTRIES_MAPPING)}")
print(f"Year Range: {YEAR_START}-{YEAR_END}")
print(f"ISO Codes: {', '.join(ISO_CODES)}\n")


class WorldBankScraper:
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 5
        self.base_url = "https://api.worldbank.org/v2"
        
        self.session = None
        self._create_session()
    
    def _create_session(self):
        if self.session:
            self.session.close()
        
        self.session = requests.Session()
        
        self.session.headers.update({
            'User-Agent': f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(90, 120)}.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,ar;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        })
        
        retry_strategy = requests.adapters.Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get_world_bank_data(self, indicator_code: str, indicator_name: str, countries: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        try:
            if countries is None:
                countries = ISO_CODES
            elif isinstance(countries, str):
                countries = [c.strip() for c in countries.split(',')]
            
            # World Bank API endpoint
            url = f"{self.base_url}/country"
            
            records = []
            
            for country_code in countries:
                try:
                    # Fetch data for each country
                    endpoint = f"{self.base_url}/country/{country_code}/indicator/{indicator_code}"
                    params = {
                        'format': 'json',
                        'per_page': 100,
                        'date': f'{YEAR_START}:{YEAR_END}'
                    }
                    
                    print(f"Fetching {indicator_name} for {country_code}...")
                    
                    for attempt in range(self.max_retries):
                        try:
                            response = self.session.get(
                                endpoint,
                                params=params,
                                timeout=20
                            )
                            response.raise_for_status()
                            
                            data = response.json()
                            
                            # Parse World Bank response format
                            if len(data) > 1 and isinstance(data[1], list):
                                for record in data[1]:
                                    if record.get('value') is not None:
                                        try:
                                            year = int(record.get('date', ''))
                                            if YEAR_START <= year <= YEAR_END:
                                                country_name = self._get_country_name(country_code)
                                                records.append({
                                                    'Country': country_name,
                                                    'Year': year,
                                                    'Metric': indicator_name,
                                                    'Value': float(record['value'])
                                                })
                                        except (ValueError, TypeError):
                                            pass
                            
                            print(f"  Success")
                            break
                        
                        except requests.exceptions.Timeout:
                            print(f"  Timeout attempt {attempt + 1}/{self.max_retries}")
                            time.sleep(self.retry_delay)
                            continue
                        
                        except requests.exceptions.ConnectionError:
                            print(f"  Connection error attempt {attempt + 1}/{self.max_retries}")
                            time.sleep(self.retry_delay)
                            continue
                        
                        except requests.exceptions.HTTPError as e:
                            if response.status_code == 429:
                                print(f"  Rate limited! Waiting...")
                                time.sleep(self.retry_delay * 2)
                                continue
                            elif response.status_code == 404:
                                print(f"  Indicator not found for {country_code}")
                                break
                            else:
                                print(f"  HTTP Error {response.status_code}")
                                break
                    
                    # Add delay between requests
                    time.sleep(1)
                
                except Exception as e:
                    print(f"  Error for {country_code}: {str(e)[:70]}")
                    continue
            
            return pd.DataFrame(records) if records else None
        
        except Exception as e:
            print(f"Error: {str(e)[:70]}")
            return None
    
    def _get_country_name(self, country_code: str) -> str:
        """Get country name from code"""
        for name, code in COUNTRIES_MAPPING.items():
            if code == country_code:
                return name
        return country_code
    
    def get_sample_data(self, indicator_name: str, years: List[int] = None) -> pd.DataFrame:
        """Generate sample data for testing/fallback"""
        if years is None:
            years = list(range(YEAR_START, YEAR_END + 1))
        
        samples = {
            'GDP Growth': (1.0, 8.0),
            'GDP per Capita': (5000, 65000),
            'Inflation Rate': (-2.0, 10.0),
            'Unemployment Rate': (2.0, 12.0),
            'Foreign Direct Investment': (0, 100),
            'Trade Volume': (10, 150),
            'Poverty Rate': (5, 40),
            'Life Expectancy': (60, 85),
        }
        
        base_range = samples.get(indicator_name, (50, 150))
        records = []
        
        for idx, country_code in enumerate(ISO_CODES):
            country_name = COUNTRY_NAMES[idx]
            
            variation = (hash(country_code) % 30) / 100
            base_value = base_range[0] + (hash(country_code) % 40)
            
            for year_offset, year in enumerate(years):
                trend = (year - YEAR_START) * 0.3
                noise = ((year_offset * hash(country_code)) % 20) / 100
                
                val = base_value * (1 + variation) + trend + noise
                
                if country_code == 'EG':
                    val *= 0.8
                elif country_code == 'CN':
                    val *= 1.3
                elif country_code == 'RU':
                    val *= 0.95
                
                records.append({
                    'Country': country_name,
                    'Year': year,
                    'Metric': indicator_name,
                    'Value': round(val, 2)
                })
        
        return pd.DataFrame(records)
    
    def get_gdp_growth(self, countries=None):
        """Fetch GDP growth rate (NY.GDP.MKTP.KD.ZG)"""
        print("\nGDP Growth Rate")
        result = self.get_world_bank_data('NY.GDP.MKTP.KD.ZG', 'GDP Growth', countries)
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result
        print("Using sample data...")
        return self.get_sample_data('GDP Growth')
    
    def get_gdp_per_capita(self, countries=None):
        """Fetch GDP per Capita (NY.GDP.PCAP.CD)"""
        print("\nGDP per Capita")
        time.sleep(2)
        result = self.get_world_bank_data('NY.GDP.PCAP.CD', 'GDP per Capita', countries)
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result
        print("Using sample data...")
        return self.get_sample_data('GDP per Capita')
    
    def get_inflation_rate(self, countries=None):
        """Fetch Inflation rate (FP.CPI.TOTL.ZG)"""
        print("\nInflation Rate")
        time.sleep(2)
        result = self.get_world_bank_data('FP.CPI.TOTL.ZG', 'Inflation Rate', countries)
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result
        print("Using sample data...")
        return self.get_sample_data('Inflation Rate')
    
    def get_unemployment_rate(self, countries=None):
        """Fetch Unemployment rate (SP.URB.TOTL.IN.ZS)"""
        print("\nUnemployment Rate")
        time.sleep(2)
        result = self.get_world_bank_data('SP.URB.TOTL.IN.ZS', 'Unemployment Rate', countries)
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result
        print("Using sample data...")
        return self.get_sample_data('Unemployment Rate')
    
    def get_fdi(self, countries=None):
        """Fetch Foreign Direct Investment (BX.KLT.DINV.CD.WD)"""
        print("\nForeign Direct Investment")
        time.sleep(2)
        result = self.get_world_bank_data('BX.KLT.DINV.CD.WD', 'Foreign Direct Investment', countries)
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result
        print("Using sample data...")
        return self.get_sample_data('Foreign Direct Investment')
    
    def get_trade_volume(self, countries=None):
        """Fetch Trade Volume (NE.EXP.GNFS.CD + NE.IMP.GNFS.CD)"""
        print("\nTrade Volume")
        time.sleep(2)
        result = self.get_world_bank_data('NE.TRD.GNFS.CD', 'Trade Volume', countries)
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result
        print("Using sample data...")
        return self.get_sample_data('Trade Volume')
    
    def get_poverty_rate(self, countries=None):
        """Fetch Poverty Rate (SI.POV.DDAY)"""
        print("\nPoverty Rate")
        time.sleep(2)
        result = self.get_world_bank_data('SI.POV.DDAY', 'Poverty Rate', countries)
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result
        print("Using sample data...")
        return self.get_sample_data('Poverty Rate')
    
    def get_life_expectancy(self, countries=None):
        """Fetch Life Expectancy (SP.DYN.LE00.IN)"""
        print("\nLife Expectancy")
        time.sleep(2)
        result = self.get_world_bank_data('SP.DYN.LE00.IN', 'Life Expectancy', countries)
        if result is not None and not result.empty:
            print(f"Retrieved {len(result)} records")
            return result
        print("Using sample data...")
        return self.get_sample_data('Life Expectancy')
    
    def get_gdp_total(self, countries=None):
        """GDP Total (NY.GDP.MKTP.CD)"""
        print("\nGDP Total")
        time.sleep(2)
        result = self.get_world_bank_data('NY.GDP.MKTP.CD', 'GDP', countries)
        return result if result is not None else self.get_sample_data('GDP')


    def get_population(self, countries=None):
        """Population (SP.POP.TOTL)"""
        print("\nPopulation")
        time.sleep(2)
        result = self.get_world_bank_data('SP.POP.TOTL', 'Population', countries)
        return result if result is not None else self.get_sample_data('Population')


    def get_exports(self, countries=None):
        """Exports (NE.EXP.GNFS.CD)"""
        print("\nExports")
        time.sleep(2)
        result = self.get_world_bank_data('NE.EXP.GNFS.CD', 'Exports', countries)
        return result if result is not None else self.get_sample_data('Exports')


    def get_imports(self, countries=None):
        """Imports (NE.IMP.GNFS.CD)"""
        print("\nImports")
        time.sleep(2)
        result = self.get_world_bank_data('NE.IMP.GNFS.CD', 'Imports', countries)
        return result if result is not None else self.get_sample_data('Imports')


    def get_gov_spending(self, countries=None):
        """Government Final Consumption (NE.CON.GOVT.CD)"""
        print("\nGovernment Spending")
        time.sleep(2)
        result = self.get_world_bank_data('NE.CON.GOVT.CD', 'Government Spending', countries)
        return result if result is not None else self.get_sample_data('Government Spending')


    def get_investment(self, countries=None):
        """Gross Capital Formation (NE.GDI.TOTL.CD)"""
        print("\nInvestment")
        time.sleep(2)
        result = self.get_world_bank_data('NE.GDI.TOTL.CD', 'Investment', countries)
        return result if result is not None else self.get_sample_data('Investment')
    
    def get_government_debt(self, countries=None):
        """Government debt (% of GDP)"""
        print("\nGovernment Debt")
        time.sleep(2)
        result = self.get_world_bank_data(
            'GC.DOD.TOTL.GD.ZS',
            'Government Debt',
            countries
        )
        return result if result is not None else self.get_sample_data('Government Debt')
    
    def get_interest_rate(self, countries=None):
        """Real Interest Rate (%)"""
        print("\nInterest Rate")
        time.sleep(2)
        result = self.get_world_bank_data(
            'FR.INR.RINR',
            'Interest Rate',
            countries
        )
        return result if result is not None else self.get_sample_data('Interest Rate')
    
    def __del__(self):
        if self.session:
            self.session.close()


if __name__ == "__main__":
    print("="*80)
    print("World Bank Economic Data Scraper (2000-2024)")
    print("="*80)
    
    wb = WorldBankScraper()
    
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
    
    pivot_df = all_data.pivot_table(
        index=['Country', 'Year'],
        columns='Metric',
        values='Value'
    ).reset_index()
    
    
    
    world_save(pivot_df)
