from datetime import date, datetime
from typing import List


def iso_week_to_timestamp(iso_week_str: str) -> str:
    """Convert CC-MAIN-YYYY-WW (ISO week) to YYYYMMDD for the Monday of that week."""
    parts = iso_week_str.split('-')

    year = int(parts[-2])
    week = int(parts[-1])

    # ISO week: Monday is day 1
    monday_date = date.fromisocalendar(year, week, 1)
    return monday_date.strftime('%Y%m%d')

def month_to_CC_slice(month_str: str, local_path: str = 'collinfo.json') -> str:
    """Convert YYYY-MM to CC slice name: CC-MAIN-YYYY-WW."""
    url = 'https://index.commoncrawl.org/collinfo.json'

    with open(local_path, 'r') as f:
        indices = json.load(f)

    dt = datetime.strptime(month_str, '%Y-%m')
    month_name = dt.strftime('%B')
    year = str(dt.year)

    for index in indices:
        name = index['name']
        if month_name in name and year in name:
            return index['id']

    raise ValueError(f'No CC slice found for month {month_str}')


def interval_to_CC_slices(start_month: str, end_month: str) -> List[str]:
    """Get list of CC slice names for months in [start_month, end_month]."""
    start_dt = datetime.strptime(start_month, '%B %Y')
    end_dt = datetime.strptime(end_month, '%B %Y')

    slices = []
    current_dt = start_dt
    while current_dt <= end_dt:
        month_str = current_dt.strftime('%Y-%m')
        cc_slice = month_to_CC_slice(month_str)
        slices.append(cc_slice)
        if current_dt.month == 12:
            current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
            current_dt = current_dt.replace(month=current_dt.month + 1)
    return slices