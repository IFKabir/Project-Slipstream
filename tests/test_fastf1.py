"""
Quick smoke test: verify FastF1 can fetch 2024 Season Race 1 data.
Run: python -m pytest tests/ -v
"""
import fastf1
import os

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "f1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def test_fastf1_2024_race1():
    session = fastf1.get_session(2024, 1, 'R')
    session.load(telemetry=False, weather=False)
    results = session.results[['Abbreviation', 'Position', 'ClassifiedPosition']]
    assert len(results) > 0, "No results were loaded"
    print(results)
