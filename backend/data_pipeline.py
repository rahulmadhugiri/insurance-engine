"""Generate mock subcontractor dataset with categorical IDs for NCF recall."""
import json
import random
from pathlib import Path

TRADES = {
    1: "Framing",
    2: "Plumbing",
    3: "Electrical",
    4: "Concrete",
    5: "Roofing",
    6: "HVAC",
    7: "Masonry",
    8: "Drywall",
    9: "Excavation",
    10: "Painting",
}

CERTIFICATIONS = {
    1: "OSHA-10",
    2: "OSHA-30",
    3: "Union",
    4: "Minority-Owned",
    5: "Woman-Owned",
    6: "EMR-Below-1.0",
}

COMPANIES = [
    "ABC Construction",
    "Smith & Sons Electric",
    "Pacific Plumbing",
    "Metro HVAC",
    "Summit Roofing",
    "Valley Drywall",
    "Northern Concrete",
    "Elite Framing",
    "Premier Painting",
    "Steel & Beam",
    "Green Earth",
    "Swift Demolition",
    "Precision Glass",
    "Heritage Masonry",
    "Apex Flooring",
    "Coastal Insulation",
    "Urban Excavation",
    "Prime Electrical",
    "Riverside Paving",
    "Mountain Lumber",
]

SUFFIXES = ["Inc", "LLC", "Co", "Corp", "Ltd", "Group"]


def generate_subcontractors(count: int = 150) -> list[dict]:
    """Generate subcontractors with discrete integer IDs for embedding-based model inputs."""
    subcontractors = []
    for sub_id in range(1, count + 1):
        base = random.choice(COMPANIES)
        suffix = random.choice(SUFFIXES)
        serial = f" #{random.randint(1, 999)}" if sub_id > len(COMPANIES) else ""
        name = f"{base} {suffix}{serial}"

        primary_trade_id = random.randint(1, len(TRADES))
        certification_id = random.randint(1, len(CERTIFICATIONS))

        subcontractors.append(
            {
                "sub_id": sub_id,
                "name": name,
                "primary_trade_id": primary_trade_id,
                "certification_id": certification_id,
                "headcount": random.randint(10, 500),
                "years_in_business": random.randint(1, 40),
            }
        )

    return subcontractors


def main():
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "subcontractors.json"
    data = generate_subcontractors()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} subcontractors -> {output_path}")


if __name__ == "__main__":
    main()
