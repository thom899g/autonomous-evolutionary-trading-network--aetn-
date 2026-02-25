"""
Data ingestion module for fetching, processing, and caching market data.
Supports multiple exchanges via CCXT with robust error handling.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from loguru import logger
import ccxt
from dataclasses import dataclass
import hashlib
import json

from config import config
from database.firestore_client import FirestoreClient

@dataclass
class MarketData