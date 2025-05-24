from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import uvicorn
from datetime import datetime
import json
import logging
from typing import Dict, Set

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Initialization
def init_db():
    with sqlite3.connect('waste.db') as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            item_type TEXT CHECK(item_type IN ('organic')),
            confidence FLOAT
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON detections(detection_time)")
        logger.info("Database initialized")

init_db()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.logger = logging.getLogger("websocket")

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"New connection established. Total: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.logger.info(f"Connection removed. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.warning(f"Broadcast failed: {e}")
                disconnected.add(connection)
        
        for connection in disconnected:
            await self.disconnect(connection)

manager = ConnectionManager()

@app.post("/api/organic")
async def add_detection(confidence: float = Query(..., gt=0, le=1)):
    logger.info(f"New detection with confidence: {confidence}")
    
    with sqlite3.connect('waste.db') as conn:
        cursor = conn.execute(
            "INSERT INTO detections (item_type, confidence) VALUES (?, ?) RETURNING id, detection_time",
            ('organic', confidence)
        )
        detection_id, detection_time = cursor.fetchone()
    
    stats = get_stats()
    response_data = {
        "event": "new_detection",
        "data": {
            "id": detection_id,
            "confidence": confidence,
            "timestamp": detection_time
        },
        "stats": stats
    }
    
    await manager.broadcast(response_data)
    return response_data

@app.get("/api/stats")
def get_stats(hours: int = 24):
    with sqlite3.connect('waste.db') as conn:
        daily = conn.execute("""
            SELECT COUNT(*), AVG(confidence) 
            FROM detections 
            WHERE DATE(detection_time) = DATE('now')
        """).fetchone()
        
        hourly = conn.execute("""
            SELECT strftime('%H', detection_time) as hour, 
                   COUNT(*) as count,
                   AVG(confidence) as avg_confidence
            FROM detections
            WHERE detection_time > datetime('now', ?)
            GROUP BY hour
        """, (f"-{hours} hours",)).fetchall()
    
    return {
        "daily_total": daily[0] or 0,
        "daily_avg_confidence": float(daily[1] or 0),
        "compost_kg": (daily[0] or 0) * 0.01,
        "hourly_data": [
            {"hour": h, "count": c, "confidence": float(conf or 0)} 
            for h, c, conf in hourly
        ]
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket) 
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received: {data}")
    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        await manager.disconnect(websocket)
        
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ws_ping_interval=20, 
        ws_ping_timeout=20
    )