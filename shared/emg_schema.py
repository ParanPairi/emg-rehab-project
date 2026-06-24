from pydantic import BaseModel
from typing import Optional

class EmgPayload(BaseModel):
    type: str = "emg"
    time: float
    device_id: str
    time_ms: Optional[int] = None
    values: list[int]   # [ch0, ch1, ch2, ch3, ch4]

class DeviceStatusPayload(BaseModel):
    type: str  # "status" | "start" | "stop"
    device_id: str
    measuring: Optional[bool] = None
    time: Optional[float] = None