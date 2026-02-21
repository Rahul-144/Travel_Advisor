import requests
import json
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from serpapi import GoogleSearch
import datetime
import asyncio
from .MCP_Client import call_mcp_tool
import os
from pprint import pprint
import nest_asyncio

# Apply the patch once at the start of your script
nest_asyncio.apply()
@tool
def get_current_date() -> str:
    """Get the current local date and time."""
    now = datetime.datetime.now()
    today = datetime.date.today()
    formatted_date = today.strftime("%Y-%m-%d")
    return {"success": True, "date": formatted_date}
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




@tool
def search_hotels(location: str, check_in: str, check_out: str) -> str:
    """Search hotels using Google Hotels engine. Returns top 5 hotels with essential information."""

    async def run():
        params = {
            "engine": "google_hotels",
            "q": location,
            "check_in_date": check_in,
            "check_out_date": check_out,
            "currency": "INR"
        }

        return await call_mcp_tool("search", {
            "params": params,
            "mode": "compact"
        })
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        raw_results = loop.run_until_complete(run())
    finally:
        loop.close()
    
    # MCP returns a list of text strings containing JSON
    # Parse the first item which contains the actual JSON data
    if raw_results and isinstance(raw_results, list):
        data = json.loads(raw_results[0])
        hotels_list = data.get("properties", [])
    else:
        return json.dumps({"hotels": [], "error": "No results found"})
    
    # Extract and clean hotel data
    cleaned_hotels = []
    
    for hotel in hotels_list[:5]:  # Limit to top 5 hotels
        cleaned_hotel = {
            "name": hotel.get("name", "N/A"),
            "type": hotel.get("type", "hotel"),
            "rating": hotel.get("overall_rating", "N/A"),
            "reviews_count": hotel.get("reviews", 0),
            "price_per_night": hotel.get("rate_per_night", {}).get("lowest", "N/A"),
            "total_price": hotel.get("total_rate", {}).get("lowest", "N/A"),
            "link": hotel.get("link", "N/A"),
            "amenities": hotel.get("amenities", [])[:8],  # Limit amenities
            "description": hotel.get("description", ""),
            "check_in_time": hotel.get("check_in_time", "N/A"),
            "check_out_time": hotel.get("check_out_time", "N/A"),
        }
        
        # Add location rating if available
        if "location_rating" in hotel:
            cleaned_hotel["location_rating"] = hotel["location_rating"]
        
        cleaned_hotels.append(cleaned_hotel)
    
    # Return as formatted JSON string
    return {"hotels": cleaned_hotels}
