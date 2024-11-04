from datetime import datetime, timedelta

def get_next_business_days(start_date, num_days=5):
    """Generate the next business days, skipping Saturdays and Sundays."""
    business_days = []
    current_date = start_date
    while len(business_days) < num_days:
        if current_date.weekday() < 5:  # Monday=0, Sunday=6
            business_days.append(current_date)
        current_date += timedelta(days=1)
    return business_days
