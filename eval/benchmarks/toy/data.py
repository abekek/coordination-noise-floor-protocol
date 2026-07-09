"""Hand-coded inventory and queries for the MiniTrip toy benchmark."""

from __future__ import annotations

# Flights: 8 records across 4 city pairs (NYC↔LON, NYC↔PAR, LON↔TOK, PAR↔TOK).
# `available_in_seeds` controls which trials see this flight as bookable.
FLIGHTS: list[dict] = [
    {"id": "F1", "origin": "NYC", "destination": "LON", "airline": "Air1",
     "price": 350, "available_in_seeds": [42, 7, 13]},
    {"id": "F2", "origin": "NYC", "destination": "LON", "airline": "Air2",
     "price": 280, "available_in_seeds": [42]},  # cheaper but limited
    {"id": "F3", "origin": "LON", "destination": "NYC", "airline": "Air1",
     "price": 360, "available_in_seeds": [42, 7]},
    {"id": "F4", "origin": "NYC", "destination": "PAR", "airline": "Air3",
     "price": 410, "available_in_seeds": [42, 13]},
    {"id": "F5", "origin": "PAR", "destination": "NYC", "airline": "Air3",
     "price": 420, "available_in_seeds": [42, 13]},
    {"id": "F6", "origin": "LON", "destination": "TOK", "airline": "Air4",
     "price": 600, "available_in_seeds": [42]},
    {"id": "F7", "origin": "TOK", "destination": "LON", "airline": "Air4",
     "price": 590, "available_in_seeds": [42]},
    {"id": "F8", "origin": "PAR", "destination": "TOK", "airline": "Air5",
     "price": 550, "available_in_seeds": [42, 7]},
]

# Hotels: 6 per destination × 4 destinations = 24 records.
HOTELS: list[dict] = [
    # London
    *[{"id": f"H_LON_{i}", "city": "LON", "name": f"London Hotel {i}",
       "nightly_rate": 80 + i * 30, "available_in_seeds": [42, 7, 13]}
      for i in range(1, 7)],
    # New York
    *[{"id": f"H_NYC_{i}", "city": "NYC", "name": f"NYC Hotel {i}",
       "nightly_rate": 100 + i * 35, "available_in_seeds": [42, 7]}
      for i in range(1, 7)],
    # Paris
    *[{"id": f"H_PAR_{i}", "city": "PAR", "name": f"Paris Hotel {i}",
       "nightly_rate": 90 + i * 25, "available_in_seeds": [42, 13]}
      for i in range(1, 7)],
    # Tokyo
    *[{"id": f"H_TOK_{i}", "city": "TOK", "name": f"Tokyo Hotel {i}",
       "nightly_rate": 110 + i * 30, "available_in_seeds": [42]}
      for i in range(1, 7)],
]

# Queries: 30 records, varied origin/destination/budget, three difficulty tiers.
QUERIES: list[dict] = (
    # Easy: budget ≥ cheapest possible combination × 1.5
    [{"id": f"q_easy_{i:03d}", "origin": "NYC", "destination": "LON",
      "budget": 800, "difficulty": "easy"} for i in range(1, 6)]
    + [{"id": f"q_easy_{i:03d}", "origin": "NYC", "destination": "PAR",
        "budget": 900, "difficulty": "easy"} for i in range(6, 11)]
    +
    # Medium: budget ≈ middle option's total
    [{"id": f"q_medium_{i:03d}", "origin": "NYC", "destination": "LON",
      "budget": 550, "difficulty": "medium"} for i in range(1, 6)]
    + [{"id": f"q_medium_{i:03d}", "origin": "PAR", "destination": "TOK",
        "budget": 800, "difficulty": "medium"} for i in range(6, 11)]
    +
    # Hard: budget forces the cheapest option, leaves no slack
    [{"id": f"q_hard_{i:03d}", "origin": "NYC", "destination": "LON",
      "budget": 460, "difficulty": "hard"} for i in range(1, 6)]
    + [{"id": f"q_hard_{i:03d}", "origin": "LON", "destination": "TOK",
        "budget": 750, "difficulty": "hard"} for i in range(6, 11)]
)


def get_query(query_id: str) -> dict:
    for q in QUERIES:
        if q["id"] == query_id:
            return q
    raise KeyError(f"unknown query_id {query_id!r}")
