import requests
import json
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel
from amadeus import Client, ResponseError
from dotenv import load_dotenv
from serpapi import GoogleSearch
from datetime import datetime
import os
@tool
def get_location_by_ip() -> Dict[str, Any]:
    """Get geographical location based on IP address."""
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        response.raise_for_status()
        data = response.json()
        return {
            "success": True,
            "ip": data.get("ip"),
            "city": data.get("city"),
            "region": data.get("region"),
            "country": data.get("country")
        }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


load_dotenv()

@tool
def search_flights(start: str, end: str, date: str) -> Dict[str, Any]:
    """Search available flights between two airports."""

    params = {
        "engine": "google_flights",
        "departure_id": start,
        "arrival_id": end,
        "currency": "INR",
        "type": "2",
        "outbound_date": date,
        "api_key": os.getenv("SERP_API_KEY")
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    best_flights = results.get("best_flights", [])

    cleaned_results = []

    for flight in best_flights[:3]:  # limit results
        first_leg = flight["flights"][0]

        cleaned_results.append({
            "airline": first_leg["airline"],
            "flight_number": first_leg["flight_number"],
            "departure_time": first_leg["departure_airport"]["time"],
            "arrival_time": first_leg["arrival_airport"]["time"],
            "duration_minutes": flight["total_duration"],
            "price_inr": flight["price"],
            "type": flight["type"]
        })

    return {"flights": cleaned_results}

# flight=search_flights('COK','GOI' , '2026-03-03')
# print(flight)
