import json
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Optional

import httpx


def _parse_iso_dt(value: str | None) -> Optional[datetime]:
    """
    Parse ISO8601 datetime string into aware UTC datetime.
    Accepts trailing "Z".
    """
    if not value:
        return None
    try:
        cleaned = value
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass
class SnowEvent:
    event_time: datetime
    payload: Dict[str, Any]
    photo_bytes: bytes | None


class EventMerger:
    """
    Keeps snow events in memory and merges them with ANPR events
    when the plate camera webhook arrives (snow -> plate order).
    """

    def __init__(
        self,
        upstream_url: str,
        window_seconds: int = 30,
        ttl_seconds: int = 60,
    ):
        self.upstream_url = upstream_url
        self.window = timedelta(seconds=window_seconds)
        self.ttl = timedelta(seconds=ttl_seconds)
        self._snow_events: Deque[SnowEvent] = deque()
        self._lock = threading.Lock()

    def _cleanup(self, now: datetime) -> None:
        while self._snow_events:
            oldest = self._snow_events[0]
            if now - oldest.event_time <= self.ttl:
                break
            self._snow_events.popleft()

    def add_snow_event(self, payload: Dict[str, Any], photo_bytes: bytes | None) -> None:
        event_time = (
            _parse_iso_dt(str(payload.get("event_time")))
            or datetime.now(tz=timezone.utc)
        )
        snow_payload = dict(payload)
        snow_payload["event_time"] = event_time.isoformat()
        with self._lock:
            self._cleanup(datetime.now(tz=timezone.utc))
            self._snow_events.append(SnowEvent(event_time, snow_payload, photo_bytes))
        print(
            f"[MERGER] stored snow event at {event_time.isoformat()}, "
            f"queue size={len(self._snow_events)}"
        )

    def _pop_match(self, anpr_time: datetime) -> Optional[SnowEvent]:
        best_idx = None
        best_delta = None

        for idx, snow_event in enumerate(self._snow_events):
            delta = anpr_time - snow_event.event_time
            if delta < timedelta(0):
                # snow should happen before plate; skip future events
                continue
            if delta <= self.window:
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_idx = idx

        if best_idx is None:
            return None

        # remove matched event
        match = self._snow_events[best_idx]
        del self._snow_events[best_idx]
        return match

    async def combine_and_send(
        self,
        anpr_event: Dict[str, Any],
        detection_bytes: bytes | None,
        feature_bytes: bytes | None,
        license_bytes: bytes | None,
    ) -> Dict[str, Any]:
        """
        Merge ANPR event with the closest earlier snow event (within window)
        and send a single multipart request upstream.
        """
        now = datetime.now(tz=timezone.utc)
        anpr_time = (
            _parse_iso_dt(str(anpr_event.get("event_time")))
            or now
        )

        with self._lock:
            self._cleanup(now)
            snow_event = self._pop_match(anpr_time)

        combined_event = dict(anpr_event)
        if snow_event:
            combined_event.update(
                {
                    "snow_event_time": snow_event.event_time.isoformat(),
                    "snow_camera_id": snow_event.payload.get("camera_id"),
                    "snow_volume_percentage": snow_event.payload.get(
                        "snow_volume_percentage"
                    ),
                    "snow_volume_confidence": snow_event.payload.get(
                        "snow_volume_confidence"
                    ),
                    "snow_direction_ai": snow_event.payload.get("snow_direction_ai"),
                    "matched_snow": True,
                }
            )
            if "snow_gemini_raw" in snow_event.payload:
                combined_event["snow_gemini_raw"] = snow_event.payload["snow_gemini_raw"]
        else:
            combined_event["matched_snow"] = False

        data = {"event": json.dumps(combined_event, ensure_ascii=False)}
        files = []

        if detection_bytes:
            files.append(
                ("photos", ("detectionPicture.jpg", detection_bytes, "image/jpeg"))
            )
        if feature_bytes:
            files.append(
                ("photos", ("featurePicture.jpg", feature_bytes, "image/jpeg"))
            )
        if license_bytes:
            files.append(
                ("photos", ("licensePlatePicture.jpg", license_bytes, "image/jpeg"))
            )
        if snow_event and snow_event.photo_bytes:
            files.append(
                ("photos", ("snowSnapshot.jpg", snow_event.photo_bytes, "image/jpeg"))
            )

        result = {
            "sent": False,
            "status": None,
            "error": None,
            "matched_snow": bool(snow_event),
        }

        if not self.upstream_url:
            result["error"] = "UPSTREAM_URL is empty"
            print(f"[MERGER] {result['error']}")
            return result

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.upstream_url,
                    data=data,
                    files=files or None,
                )
            result["sent"] = resp.is_success
            result["status"] = resp.status_code
            if not resp.is_success:
                result["error"] = resp.text[:400]
            print(
                f"[MERGER] upstream sent={result['sent']} "
                f"status={result['status']} matched_snow={result['matched_snow']}"
            )
        except Exception as e:
            result["error"] = str(e)
            print(f"[MERGER] error while sending event: {e}")

        return result


_merger_instance: EventMerger | None = None


def init_merger(
    upstream_url: str,
    window_seconds: int = 30,
    ttl_seconds: int = 60,
) -> EventMerger:
    """
    Initialize (or return existing) EventMerger singleton.
    """
    global _merger_instance
    if _merger_instance is None:
        _merger_instance = EventMerger(
            upstream_url=upstream_url,
            window_seconds=window_seconds,
            ttl_seconds=ttl_seconds,
        )
    return _merger_instance
