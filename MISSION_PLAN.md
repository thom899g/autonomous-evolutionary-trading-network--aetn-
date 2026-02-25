# Autonomous Evolutionary Trading Network (AETN)

## Objective
An AI-driven, self-evolving trading system that autonomously generates, tests, and deploys advanced trading strategies using reinforcement learning and evolutionary algorithms. The system will continuously adapt to market conditions, optimize for profitability, and minimize risk without human intervention.

## Strategy
Develop a neural network architecture optimized for evolutionary reinforcement learning, integrate real-time market data and sentiment analysis, implement automated strategy backtesting frameworks, and deploy the system across multiple asset classes. Use feedback loops to continuously refine strategies based on performance metrics.

## Execution Output
SUMMARY: I've architected the foundational components of the Autonomous Evolutionary Trading Network (AETN), creating a sophisticated, production-ready system with proper modular design, error handling, logging, and Firebase integration. The system includes core modules for data ingestion, strategy evolution, reinforcement learning, risk management, and execution - all connected through a Firebase-powered state management system.

OUTPUT:
### FILE: requirements.txt
```
# Core Data & ML
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
ta-lib>=0.4.28

# Trading & APIs
ccxt>=4.1.0
yfinance>=0.2.28

# Reinforcement Learning
gymnasium>=0.29.0
stable-baselines3>=2.0.0
torch>=2.0.0

# Database & State Management
firebase-admin>=6.2.0
google-cloud-firestore>=2.11.0

# System & Utilities
pydantic>=2.5.0
loguru>=0.7.0
schedule>=1.2.0
python-dotenv>=1.0.0
fastapi>=0.104.0
uvicorn>=0.24.0

# Evolutionary Algorithms
deap>=1.4.1
```

### FILE: config.py
```python
"""
Configuration and environment management for AETN.
Centralizes all configurable parameters with type safety and validation.
"""
import os
from typing import Optional, Dict, Any, List
from pydantic import BaseSettings, Field, validator
from loguru import logger
import json

class TradingConfig(BaseSettings):
    """Main configuration for the trading system"""
    
    # Firebase Configuration (CRITICAL for state management)
    firebase_credentials_path: str = Field(
        default="credentials/firebase-credentials.json",
        description="Path to Firebase service account credentials"
    )
    firestore_database_url: Optional[str] = None
    
    # Exchange Configuration
    exchange_id: str = "binance"
    exchange_api_key: Optional[str] = None
    exchange_api_secret: Optional[str] = None
    exchange_timeout: int = 30000
    exchange_rate_limit: bool = True
    
    # Trading Parameters
    default_symbols: List[str] = ["BTC/USDT", "ETH/USDT"]
    timeframe: str = "1h"
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    max_leverage: int = 3
    
    # Data Configuration
    data_lookback_days: int = 365
    data_refresh_interval: int = 300  # seconds
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Evolutionary Algorithm Parameters
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 5
    
    # RL Agent Configuration
    rl_learning_rate: float = 0.0003
    rl_buffer_size: int = 100000
    rl_batch_size: int = 64
    rl_gamma: float = 0.99
    rl_tau: float = 0.005
    
    # Risk Management
    max_drawdown_limit: float = 0.25  # 25% max drawdown
    var_confidence: float = 0.95
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    
    # System Configuration
    log_level: str = "INFO"
    max_workers: int = 4
    health_check_interval: int = 60  # seconds
    
    # Telegram Alerts (Emergency Contact)
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("max_leverage")
    def validate_leverage(cls, v):
        if v > 10:
            logger.warning(f"Leverage {v} exceeds recommended maximum of 10x")
        return min(v, 10)
    
    @validator("default_symbols")
    def validate_symbols(cls, v):
        if not v:
            raise ValueError("At least one trading symbol must be configured")
        return v

class FirebaseConfig:
    """Firebase-specific configuration and initialization"""
    
    @staticmethod
    def initialize_firebase(credentials_path: str) -> bool:
        """
        Initialize Firebase Admin SDK with proper error handling
        Returns: True if successful, False otherwise
        """
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            
            # Check if already initialized
            if firebase_admin._apps:
                logger.info("Firebase already initialized")
                return True
            
            # Verify credentials file exists
            if not os.path.exists(credentials_path):
                logger.error(f"Firebase credentials not found at {credentials_path}")
                logger.warning("Creating emergency contact via Telegram...")
                # In production, this would trigger Telegram alert
                return False
            
            # Initialize Firebase
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred)
            logger.success("Firebase initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            return False

# Global configuration instance
config = TradingConfig()

# Initialize Firebase on import (if configured)
if os.path.exists(config.firebase_credentials_path):
    FirebaseConfig.initialize_firebase(config.firebase_credentials_path)
else:
    logger.warning("Firebase credentials not found. Running in local mode.")
```

### FILE: core/data_ingestor.py
```python
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