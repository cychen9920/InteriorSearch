import argparse
import csv
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path


STARTER_IMAGES = [
    {
        "filename": "kitchen_interior_design.jpg",
        "commons_file": "Kitchen interior design.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Kitchen_interior_design.jpg",
        "credit": "EddieRider at English Wikipedia",
        "license": "Public domain",
    },
    {
        "filename": "living_room_modern.jpg",
        "commons_file": "Living Room.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Living_Room.jpg",
        "credit": "Tim Collins",
        "license": "CC BY-SA 3.0",
    },
    {
        "filename": "living_room_bottomley_house.jpg",
        "commons_file": "Living Room (3650655376).jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Living_Room_(3650655376).jpg",
        "credit": "Steve Snodgrass",
        "license": "CC BY-SA 2.0",
    },
    {
        "filename": "pope_leighey_living_room.jpg",
        "commons_file": "Pope-Leighey House - Living room interior - HABS VA,30-FALCH,2-19.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Pope-Leighey_House_-_Living_room_interior_-_HABS_VA,30-FALCH,2-19.jpg",
        "credit": "Library of Congress, HABS",
        "license": "Public domain",
    },
    {
        "filename": "kitchen_cups_interior.jpg",
        "commons_file": "(Interior view of kitchen (AM 81886-2).jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:(Interior_view_of_kitchen_(AM_81886-2).jpg",
        "credit": "Tudor Washington Collins",
        "license": "Public domain / no known restrictions",
    },
    {
        "filename": "apartment_kitchen_seattle.jpg",
        "commons_file": "Apartment kitchen interior, nd (SEATTLE 2545).jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Apartment_kitchen_interior,_nd_(SEATTLE_2545).jpg",
        "credit": "Unknown photographer, Seattle Municipal Archives",
        "license": "Public domain / no known restrictions",
    },
    {
        "filename": "historic_kitchen_dpla.jpg",
        "commons_file": "Kitchen - interior - DPLA - b1a85d2f759ba4a9486ff9309943668c.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Kitchen_-_interior_-_DPLA_-_b1a85d2f759ba4a9486ff9309943668c.jpg",
        "credit": "University of Illinois Urbana-Champaign University Library",
        "license": "Public domain",
    },
    {
        "filename": "hotel_bedroom_open_door.jpg",
        "commons_file": "Bedroom hotel interior with open door window. (51536308276).jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Bedroom_hotel_interior_with_open_door_window._(51536308276).jpg",
        "credit": "Marco Verch",
        "license": "CC BY 2.0",
    },
    {
        "filename": "woodbury_bedroom.jpg",
        "commons_file": "Woodbury interior bedroom - DPLA - ba97af76d2d3a6ef6612878f1ce536b1.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Woodbury_interior_bedroom_-_DPLA_-_ba97af76d2d3a6ef6612878f1ce536b1.jpg",
        "credit": "University of Colorado Boulder Libraries",
        "license": "Public domain",
    },
    {
        "filename": "harlem_river_bedroom.jpg",
        "commons_file": "Harlem River Houses Bedroom (1937).jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Harlem_River_Houses_Bedroom_(1937).jpg",
        "credit": "Federal housing documentation, 1937",
        "license": "Public domain",
    },
    {
        "filename": "dining_room_w_long.jpg",
        "commons_file": "Interior of a Dining Room, by W. Long.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Interior_of_a_Dining_Room,_by_W._Long.jpg",
        "credit": "W. Long / New York Public Library",
        "license": "Public domain",
    },
    {
        "filename": "loudoun_house_dining_room.jpg",
        "commons_file": "Loudoun House, interior; dining room, 1903.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Loudoun_House,_interior;_dining_room,_1903.jpg",
        "credit": "Daderot / William Cassius Goodloe II",
        "license": "Public domain",
    },
    {
        "filename": "government_house_dining_room.jpg",
        "commons_file": "Government House (1868-1912), interior, dining room 1.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Government_House_(1868-1912),_interior,_dining_room_1.jpg",
        "credit": "Augustus H. Oscar Freemantle / Toronto Public Library",
        "license": "Public domain",
    },
    {
        "filename": "old_faithful_dining_room.jpg",
        "commons_file": "Old Faithful Inn, dining room (9411152076).jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Old_Faithful_Inn,_dining_room_(9411152076).jpg",
        "credit": "Jim Peaco, National Park Service",
        "license": "Public domain",
    },
    {
        "filename": "boynton_house_dining_room.jpg",
        "commons_file": "Boynton House - Dining room.jpg",
        "source_page": "https://commons.wikimedia.org/wiki/File:Boynton_House_-_Dining_room.jpg",
        "credit": "Library of Congress, HABS",
        "license": "Public domain",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download real starter interior images.")
    parser.add_argument("--output-dir", default="data/images")
    parser.add_argument("--credits", default="data/image_credits.csv")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image in STARTER_IMAGES:
        output_path = output_dir / image["filename"]
        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            continue

        url = file_path_url(image["commons_file"], width=args.width)
        print(f"Downloading {image['filename']}")
        try:
            download(url, output_path)
        except Exception as exc:
            print(f"Failed to download {image['filename']}: {exc}")
            if output_path.exists() and output_path.stat().st_size == 0:
                output_path.unlink()
        time.sleep(args.delay)

    write_credits(Path(args.credits))
    print(f"Wrote credits to {Path(args.credits).resolve()}")


def file_path_url(commons_file: str, width: int) -> str:
    encoded = urllib.parse.quote(commons_file)
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded}?width={width}"


def download(url: str, output_path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "InteriorStyle/0.1 (starter image downloader)"},
    )
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with output_path.open("wb") as output_file:
                    shutil.copyfileobj(response, output_file)
            return
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(8)

    raise RuntimeError(last_error)


def write_credits(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "source_page", "credit", "license"],
        )
        writer.writeheader()
        for image in STARTER_IMAGES:
            writer.writerow(
                {
                    "filename": image["filename"],
                    "source_page": image["source_page"],
                    "credit": image["credit"],
                    "license": image["license"],
                }
            )


if __name__ == "__main__":
    main()
