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

