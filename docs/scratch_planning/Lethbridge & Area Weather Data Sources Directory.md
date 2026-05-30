## Lethbridge & Area Weather Data Sources Directory

## 1\. Federal & Historic Climate Stations

These stations are maintained by Environment Canada and provide the highest-quality, most consistent baselines for machine learning training. They are ideal for building your core historical dataset.

## Lethbridge Airport (YQL / CWQL)

* **Provider:** Environment Canada / Meteorological Service of Canada  
* **Location:** South of city limits (Lethbridge Airport)  
* **Data Frequency:** Hourly and Daily  
* **Historical Depth:** Continuous data from 1953 to present  
* **Key Parameters:** Temperature, dew point, relative humidity, wind speed/direction, station pressure, hourly precipitation, visibility, and weather descriptions.  
* **Access Portal:** Environment Canada Historical Climate Data

## Lethbridge Research Centre (Lethbridge CDA)

* **Provider:** Environment Canada / Agriculture and Agri-Food Canada  
* **Location:** East Lethbridge (Research Station)  
* **Data Frequency:** Primarily Daily (some historic hourly blocks)  
* **Historical Depth:** Massive historical archive dating back to 1908  
* **Key Parameters:** Daily maximum/minimum temperatures, daily rainfall, daily snowfall, and total precipitation.  
* **Access Portal:** Environment Canada Historical Climate Data

---

## 2\. Provincial & Agricultural Networks (ACIS)

Operated by the Alberta Climate Information Service (ACIS), these automated stations feature highly accurate tipping-bucket rain gauges. They provide crucial data points around the city perimeter and county.

## In-City / City Edge Stations

* **Lethbridge Demo Farm (AGDM):** Located on the eastern urban-rural fringe.  
* **Lethbridge Agtech (AGCM):** Located adjacent to city boundaries.

## Lethbridge County Stations (Storm Tracking Vectors)

* **Picture Butte West (AGDM):** Located directly north of Lethbridge; excellent for tracking cold fronts dropping down from the north.  
* **Iron Springs (AGDM):** Located northeast of the city.  
* **Blood Tribe (AGDM):** Located southwest of the city; critical for capturing convective summer storms rolling in from the foothills.  
* **Warner / Milk River Stations:** Located further south toward the Montana border; useful for tracking upslope systems.

## Network Details

* **Data Frequency:** Hourly and Daily  
* **Historical Depth:** Generally 10 to 30+ years of continuous digital records  
* **Key Parameters:** Air temperature, precipitation (tipping bucket), relative humidity, wind speed/direction, solar radiation, and soil temperature.  
* **Access Portal:** Alberta ACIS Interactive Map & Data Viewer

---

## 3\. Private Weather Station (PWS) Networks

Crowd-sourced networks consisting of thousands of consumer and enthusiast-grade weather stations. These allow you to capture hyper-local rainfall variants inside specific city neighborhoods (e.g., West Lethbridge, North Lethbridge, Henderson Lake).

## Weather Underground (WU)

* **Network Characteristics:** The largest global crowd-sourced network. Features dense urban coverage in Lethbridge neighborhoods using hardware like Davis Instruments and Ecowitt.  
* **Data Frequency:** Real-time updates with hourly historical rollups.  
* **Access Portal:** Weather Underground Wundermap

## Ambient Weather Network (AWN)

* **Network Characteristics:** Populated by owners of Ambient Weather consumer hardware. Multiple active stations are scattered across Lethbridge city and regional farms.  
* **Data Frequency:** Real-time and hourly historical tracking.  
* **Access Portal:** Ambient Weather Network Map

## WeatherFlow Tempest Network

* **Network Characteristics:** Rapidly growing network utilizing all-in-one smart stations. They feature haptic rain sensors that measure rain via kinetic impact vibration instead of physical buckets. Highly developer-friendly with a free public JSON API.  
* **Data Frequency:** Continuous real-time streaming and archived hourly blocks.  
* **Access Portal:** Tempest Weather Network Map

---

## 4\. Citizen-Science Manual Networks

High-accuracy manual networks that serve as the ground-truth benchmark for verifying whether automated digital sensors failed or miscalculated during a storm event.

## CoCoRaHS (Community Collaborative Rain, Hail and Snow Network)

* **Provider:** Volunteers using standardized high-capacity manual rain gauges.  
* **Location:** Multiple active human observers registered within Lethbridge city limits, Coaldale, and Taber.  
* **Data Frequency:** Daily (reported every morning, typically at 7:00 AM).  
* **Historical Depth:** Multi-year structured archives depending on individual observer longevity.  
* **Key Parameters:** 24-hour accumulated rainfall, snowfall, water equivalent of snow, and hail details.  
* **Access Portal:** [CoCoRaHS Data Portal](https://www.cocorahs.org/)

