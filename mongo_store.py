import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from bson import ObjectId
    from pymongo import ASCENDING, MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    PYMONGO_AVAILABLE = True
except Exception:
    ObjectId = None
    MongoClient = None
    Collection = Any
    Database = Any
    ASCENDING = 1
    PYMONGO_AVAILABLE = False

load_dotenv()


def _clean_doc(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_doc(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_doc(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        ts = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(ts) else pd.Timestamp(ts).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


class MongoStore:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        self.uri = uri or os.getenv("MONGODB_URI", "").strip()
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", "stocks_app").strip()
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None

        if not PYMONGO_AVAILABLE:
            return
        if not self.uri:
            return

        self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[self.db_name]
        self.client.admin.command("ping")
        self._ensure_indexes()

    @property
    def enabled(self) -> bool:
        return self.db is not None

    def _col(self, name: str) -> Collection:
        if self.db is None:
            raise RuntimeError("MongoStore is not configured. Set MONGODB_URI first.")
        return self.db[name]

    def _ensure_indexes(self) -> None:
        self._col("users").create_index([("email", ASCENDING)], unique=True, sparse=True)
        self._col("screening_runs").create_index([("run_id", ASCENDING)], unique=True)
        self._col("global_cache").create_index([("cache_key", ASCENDING)], unique=True)
        self._col("analysis_sessions").create_index([("session_key", ASCENDING)], unique=True)
        self._col("analysis_sessions").create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])
        self._col("stock_reports").create_index([("session_id", ASCENDING), ("ticker", ASCENDING)], unique=True)
        self._col("web_jobs").create_index([("job_id", ASCENDING)], unique=True)
        self._col("web_jobs").create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])

    def upsert_user(self, user_id: str, email: Optional[str] = None, password_hash: Optional[str] = None) -> None:
        doc = {
            "user_id": user_id,
            "email": email,
            "password_hash": password_hash,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if email is None:
            doc.pop("email")
        if password_hash is None:
            doc.pop("password_hash")
        self._col("users").update_one(
            {"user_id": user_id},
            {"$set": _clean_doc(doc), "$setOnInsert": {"created_at": datetime.utcnow().isoformat()}},
            upsert=True,
        )

    def upsert_global_cache(self, cache_key: str, payload: Dict[str, Any]) -> None:
        self._col("global_cache").update_one(
            {"cache_key": cache_key},
            {"$set": {"cache_key": cache_key, "payload": _clean_doc(payload), "updated_at": datetime.utcnow().isoformat()}},
            upsert=True,
        )

    def upsert_screening_run(self, run_id: str, payload: Dict[str, Any]) -> None:
        self._col("screening_runs").update_one(
            {"run_id": run_id},
            {"$set": {"run_id": run_id, **_clean_doc(payload), "updated_at": datetime.utcnow().isoformat()}},
            upsert=True,
        )

    def create_analysis_session(self, payload: Dict[str, Any]) -> str:
        doc = _clean_doc(payload)
        if "created_at" not in doc:
            doc["created_at"] = datetime.utcnow().isoformat()
        if "status" not in doc:
            doc["status"] = "created"
        result = self._col("analysis_sessions").insert_one(doc)
        return str(result.inserted_id)

    def update_analysis_session(self, session_id: str, updates: Dict[str, Any]) -> None:
        selector: Dict[str, Any]
        try:
            selector = {"_id": ObjectId(session_id)} if ObjectId is not None else {"session_key": session_id}
        except Exception:
            selector = {"session_key": session_id}
        self._col("analysis_sessions").update_one(
            selector,
            {"$set": {**_clean_doc(updates), "updated_at": datetime.utcnow().isoformat()}},
        )

    def upsert_stock_report(self, session_id: str, ticker: str, payload: Dict[str, Any]) -> None:
        self._col("stock_reports").update_one(
            {"session_id": session_id, "ticker": ticker},
            {
                "$set": {
                    "session_id": session_id,
                    "ticker": ticker,
                    **_clean_doc(payload),
                    "updated_at": datetime.utcnow().isoformat(),
                }
            },
            upsert=True,
        )


def get_mongo_store() -> Optional[MongoStore]:
    if not PYMONGO_AVAILABLE:
        print("Mongo disabled: pymongo is not installed yet.")
        return None
    try:
        store = MongoStore()
        return store if store.enabled else None
    except Exception as exc:
        print(f"Mongo disabled: {exc}")
        return None
