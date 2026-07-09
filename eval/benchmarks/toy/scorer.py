"""Hard-constraint completion check for MiniTrip."""

from __future__ import annotations


def check_completion(
    *, flight_id: str | None, hotel_id: str | None,
    flight_price: float, hotel_total: float, budget: float,
) -> bool:
    if flight_id is None or hotel_id is None:
        return False
    return (flight_price + hotel_total) <= budget
