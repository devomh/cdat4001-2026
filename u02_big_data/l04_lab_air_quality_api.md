---
title: "Lab: Air Quality from a Live API"
unit: "II"
lesson: "04"
type: lab
tags: [json, web-api, requests, normalization, duckdb]
difficulty: introductory
duration: "60 mins"
---

**Goal:** answer L01's question for real -- *which Humacao-area municipality had the worst
air quality?* -- by acquiring PM2.5 readings from a live web API, flattening the JSON, and
ranking with one GROUP BY. Pairs with the concept note
[JSON & Web APIs: Acquiring Semi-Structured Data](l04_concept_json_web_apis.qmd).

> This page is the read-only view. To run the lab, open the notebook (`l04_lab_air_quality_api.ipynb`) -- in Colab via the badge below, or locally.
>
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/devomh/cdat4001-2026/blob/main/u02_big_data/l04_lab_air_quality_api.ipynb)

## Prerequisites & Setup

Run this first. It fetches hourly PM2.5 for three municipalities from the **Open-Meteo
Air Quality API** (free, no key, education-friendly) for a **fixed date range** (March
1-14, 2024) -- past dates, so everyone gets the same values -- and saves the raw JSON to
`data/aq_march2024.json`. If the network is down, it falls back to a bundled copy of the
same data (`data/aq_fallback.json`, fetched from the same API at authoring time).

```python
# Setup: run this cell first (required for Colab -- it resets on open)
%pip install -q requests pandas duckdb==1.4.2   # duckdb pinned (API we teach); requests/pandas float -- Colab preinstalls them

import json, os, requests
import pandas as pd, duckdb

os.makedirs("data", exist_ok=True)

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MUNICIPALITIES = {                       # town-center coordinates
    "Humacao": (18.1496, -65.8276),
    "Naguabo": (18.2119, -65.7359),
    "Caguas":  (18.2341, -66.0356),
}

def fetch_pm25(lat, lon):
    """Hourly PM2.5 for March 1-14, 2024 (fixed past dates keep the answers stable)."""
    params = {
        "latitude": lat, "longitude": lon, "hourly": "pm2_5",
        "start_date": "2024-03-01", "end_date": "2024-03-14",
        "timezone": "America/Puerto_Rico",
    }
    r = requests.get(API_URL, params=params, timeout=30)
    r.raise_for_status()                 # any non-2xx status becomes a visible error
    return r.json()

try:
    raw = {name: fetch_pm25(lat, lon) for name, (lat, lon) in MUNICIPALITIES.items()}
    print("fetched live from the API")
except Exception as err:                 # offline? use the bundled copy (same dates, same values)
    print("API unreachable, using bundled fallback --", err)
    raw = {k: v for k, v in json.load(open("data/aq_fallback.json")).items()
           if k in MUNICIPALITIES}

with open("data/aq_march2024.json", "w") as f:
    json.dump(raw, f)
print("saved data/aq_march2024.json with", len(raw), "municipalities")
```

~~~text
fetched live from the API
saved data/aq_march2024.json with 3 municipalities
~~~

## Step 1: Inspect the Raw JSON

Before transforming anything, look at what the API actually sent. It parsed into plain
Python objects: a dict at the top.

```python
humacao = raw["Humacao"]
print(type(humacao))
print(list(humacao.keys()))
```

<details><summary>Expected Output</summary>

~~~text
<class 'dict'>
['latitude', 'longitude', 'generationtime_ms', 'utc_offset_seconds', 'timezone', 'timezone_abbreviation', 'elevation', 'hourly_units', 'hourly']
~~~
*(Metadata keys first -- where the server snapped our coordinates to, timing, units --
then `hourly`, the payload. The server tells you about your own request: that is the API
contract at work.)*
</details>

The payload nests one level deeper: `hourly` holds two **parallel arrays**.

```python
print(humacao["hourly"]["time"][:3])
print(humacao["hourly"]["pm2_5"][:3])
print(len(humacao["hourly"]["pm2_5"]), "hourly readings")
```

<details><summary>Expected Output</summary>

~~~text
['2024-03-01T00:00', '2024-03-01T01:00', '2024-03-01T02:00']
[3.7, 3.8, 3.9]
336 hourly readings
~~~
*(14 days x 24 hours = 336 readings. Position links the arrays: the reading at index 0
happened at the time at index 0. This shape is natural for JSON and useless for GROUP BY
-- which is why Step 3 exists.)*
</details>

## Step 2: Make a Request Yourself -- *completion problem*

The setup cell hid the request inside a function. Now build one in the open, for Naguabo.
**Uncomment and fill the two `____` blanks**, then run.

```python
# Uncomment and fill the two ____ blanks, then run:
# params = {
#     "latitude": ____,                  # Naguabo's latitude (see MUNICIPALITIES above)
#     "longitude": -65.7359,
#     "hourly": ____,                    # the variable we want, as a string (see fetch_pm25)
#     "start_date": "2024-03-01", "end_date": "2024-03-14",
#     "timezone": "America/Puerto_Rico",
# }
# r = requests.get(API_URL, params=params, timeout=30)
# print(r.status_code)
# print(r.json()["hourly"]["pm2_5"][:5])
```

<details><summary>Expected Output</summary>

~~~text
200
[7.1, 7.5, 8.3, 9.2, 9.8]
~~~
*(`200` means the contract was honored: the server understood the parameters and returned
data. Naguabo's first hours already read higher than Humacao's 3.7-3.9 -- hold that
thought for Step 3.)*
</details>

## Step 3: Flatten to a Table and Answer L01's Question

JSON nests; `GROUP BY` wants rows. Walk each municipality's parallel arrays and lay them
out flat -- municipality repeated down its rows, one reading per row.

```python
frames = []
for name, resp in raw.items():
    hourly = resp["hourly"]                          # two parallel lists -> two columns
    frames.append(pd.DataFrame({
        "municipality": name,
        "time": pd.to_datetime(hourly["time"]),
        "pm25": hourly["pm2_5"],
    }))
readings = pd.concat(frames, ignore_index=True)
print(readings.shape)
readings.head(3)
```

<details><summary>Expected Output</summary>

~~~text
(1008, 3)
  municipality                time  pm25
0      Humacao 2024-03-01 00:00:00   3.7
1      Humacao 2024-03-01 01:00:00   3.8
2      Humacao 2024-03-01 02:00:00   3.9
~~~
*(3 municipalities x 336 readings = 1008 rows. The nesting is gone; the municipality name
is repeated down its rows -- the price of flatness the concept note described.)*
</details>

Now the payoff: L01 traced this exact question through the five stages on paper.
Stage 4 is one query -- ranked, with a coverage count so we can say *how sure we are*.

```python
con = duckdb.connect()
con.sql("""
    SELECT municipality,
           ROUND(AVG(pm25), 2) AS avg_pm25,
           COUNT(*)            AS n_readings
    FROM readings
    GROUP BY municipality
    ORDER BY avg_pm25 DESC
""").df()
```

<details><summary>Expected Output</summary>

~~~text
  municipality  avg_pm25  n_readings
0      Naguabo      6.12         336
1       Caguas      5.62         336
2      Humacao      4.55         336
~~~
*(The answer to L01's question, for these two weeks: Naguabo ranks worst at 6.12
micrograms per cubic meter, Humacao best at 4.55 -- all well below common health
guidelines, and with full coverage (336 of 336 hours each), we can state it with
confidence. From fuzzy question to defensible answer: acquire, inspect, flatten, rank.)*
</details>

## Your Turn

### Exercise 1 -- Add a Fourth Municipality

**Task:** fetch Yabucoa (latitude `18.0505`, longitude `-65.8793`) with `fetch_pm25` and
re-run the ranking query over a **new** 4-municipality table named `readings4` -- leave
`readings` untouched, Exercise 2 still needs the original three. Where does Yabucoa land?
**Hint:** build its DataFrame exactly like the loop in Step 3, then
`readings4 = pd.concat([readings, ...], ignore_index=True)` and query `readings4`.

```python
# TODO: your code here -- fetch, append, re-rank
```

<details><summary>Expected Output</summary>

~~~text
  municipality  avg_pm25  n_readings
0      Naguabo      6.12         336
1       Caguas      5.62         336
2      Humacao      4.55         336
3      Yabucoa      4.55         336
~~~
*(Yabucoa ties Humacao EXACTLY -- every hourly value is identical. That is not a
coincidence: the API serves a weather-model grid of roughly 11 km cells, and both towns
snap to the same cell. The lesson: a data source's spatial resolution limits which
questions it can answer -- distinguishing these neighbors needs ground sensors, not this
model. That caveat belongs in your stage-5 report; L01 called it being honest about
coverage.)*
</details>

### Exercise 2 -- The Worst Hour of the Day

**Task:** across all three original municipalities, which **hour of the day** has the
highest average PM2.5? Show the top 3. **Hint:** `EXTRACT(hour FROM time)` gives the
hour; group by it.

```python
# TODO: your query here
```

<details><summary>Expected Output</summary>

~~~text
   hour_of_day  avg_pm25
0           16      5.61
1           15      5.60
2           17      5.57
~~~
*(Mid-afternoon, 3-5 PM, reads highest -- consistent with daytime traffic and heat-driven
chemistry. A one-line GROUP BY already produces a finding you could report.)*
</details>

## Summary

- An API response parses into **dicts and lists**; inspect its keys before transforming.
- A GET request is **endpoint + params**; check `status_code` (or `raise_for_status()`)
  before trusting the body.
- **Flatten** parallel arrays into rows (parent values repeated), and every SQL tool from
  L02 applies unchanged.
- Fixed past dates + saving the raw response = an acquisition step others can reproduce.
- Real sources have **resolution limits** (Exercise 1) -- report them, don't hide them.
