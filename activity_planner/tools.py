import requests
import json
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel
from amadeus import Client, ResponseError
from dotenv import load_dotenv
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

# Replace with your actual API Key and API Secret
AMADEUS_API_KEY = os.getenv('AMADEUS_API_KEY')
AMADEUS_API_SECRET = os.getenv('AMADEUS_API_SECRET')

# Initialize the Amadeus client
amadeus = Client(
    client_id=AMADEUS_API_KEY,
    client_secret=AMADEUS_API_SECRET
)

def search_flights(origin, destination, departure_date, adults=1):
    """
    Searches for flight offers using the Amadeus Flight Offers Search API.

    Args:
        origin (str): Origin airport IATA code (e.g., 'SYD').
        destination (str): Destination airport IATA code (e.g., 'BKK').
        departure_date (str): Departure date in YYYY-MM-DD format.
        adults (int): Number of adult passengers.

    Returns:
        list: A list of flight offers if successful, otherwise None.
    """
    try:
        print(f"Searching for flights from {origin} to {destination} on {departure_date}...")
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=departure_date,
            adults=adults
        )
        print("Flight search successful!")
        # The flight data is in the 'data' field of the response
        return response.data

    except ResponseError as error:
        print(f"Error during flight search: {error}")
        return None
if __name__ == "__main__":
    # Example parameters (replace with your desired flight details)
    origin_code = 'COK'
    destination_code = 'HYD'
    date_of_departure = '2026-01-30' # Use a future date for testing

    flights = search_flights(origin_code, destination_code, date_of_departure)

    if flights:
        print(f"Found {len(flights)} flight offers.")
        # Print details of the first flight offer
        if len(flights) > 0:
            first_flight = flights[0]
            print("\nFirst flight details:")
            print(f"Price: {first_flight['price']['total']} {first_flight['price']['currency']}")
            print("Itineraries:")
            for itinerary in first_flight['itineraries']:
                for segment in itinerary['segments']:
                    print(f"  * Departure: {segment['departure']['iataCode']} at {segment['departure']['at']}")
                    print(f"    Arrival: {segment['arrival']['iataCode']} at {segment['arrival']['at']}")
                    print(f"    Carrier: {segment['carrierCode']}")
                    print(f"    Flight Number: {segment['number']}")
