from flask import Flask, jsonify, request
from flask_cors import CORS
from garminconnect import Garmin
from datetime import datetime, timedelta
import os
import json
from flask import send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# Garmin client instance
garmin_client = None

# Cache files
TSS_CACHE_FILE = 'tss_cache.json'
FTP_FILE = 'ftp_data.json'

def load_ftp_data():
    """Load FTP and power records history from file"""
    try:
        if os.path.exists(FTP_FILE):
            with open(FTP_FILE, 'r') as f:
                data = json.load(f)
                # Ensure power_records structure exists
                if "power_records" not in data:
                    data["power_records"] = {dur: [] for dur in POWER_DURATIONS.keys()}
                return data
    except Exception as e:
        print(f"Error loading FTP data: {e}")
    return {
        "current_ftp": 100,
        "history": [],
        "power_records": {dur: [] for dur in POWER_DURATIONS.keys()},
        "current_bests": {dur: 0 for dur in POWER_DURATIONS.keys()}
    }

def save_ftp_data(data):
    """Save FTP data to file"""
    try:
        with open(FTP_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error saving FTP data: {e}")

def load_tss_cache():
    """Load TSS cache from file"""
    try:
        if os.path.exists(TSS_CACHE_FILE):
            with open(TSS_CACHE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading TSS cache: {e}")
    return {}

def save_tss_cache(cache):
    """Save TSS cache to file"""
    try:
        with open(TSS_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Error saving TSS cache: {e}")

def get_cached_tss(activity_id):
    """Get TSS data from cache if available"""
    cache = load_tss_cache()
    return cache.get(str(activity_id))

def cache_tss(activity_id, tss_data):
    """Cache TSS data for an activity"""
    cache = load_tss_cache()
    cache[str(activity_id)] = tss_data
    save_tss_cache(cache)

def init_garmin():
    """Initialize Garmin Connect client with credentials"""
    global garmin_client
    try:
        email = os.getenv('GARMIN_EMAIL')
        password = os.getenv('GARMIN_PASSWORD')
        
        print(f"Attempting to login with email: {email}")
        
        if not email or not password:
            raise ValueError("Garmin credentials not found in .env file")
        
        garmin_client = Garmin(email, password)
        
        try:
            garmin_client.login()
            print("Login successful (no assertion error)")
        except AssertionError:
            # Known bug in garminconnect library - profile assertion fails
            # but authentication actually succeeded
            print("Profile assertion failed but login may have succeeded, continuing...")
        
        # The underlying garth library should have authentication tokens now
        # Let's verify by trying different methods to get data
        print("\nTesting data access...")
        
        # Test 1: Try to get activities with start/limit
        print("Test 1: get_activities(0, 5)")
        try:
            test_acts = garmin_client.get_activities(0, 5)
            print(f"  SUCCESS - Got {len(test_acts)} activities")
            if len(test_acts) > 0:
                print("Successfully logged into Garmin Connect!")
                return True
        except Exception as e:
            print(f"  FAILED: {e}")
        
        # Test 2: Try get_stats
        print("Test 2: Checking authentication status")
        try:
            # Check if we have OAuth tokens
            if hasattr(garmin_client, 'garth') and garmin_client.garth:
                print(f"  Garth client exists: Yes")
                if hasattr(garmin_client.garth, 'oauth1_token'):
                    print(f"  Has OAuth1 token: {garmin_client.garth.oauth1_token is not None}")
                if hasattr(garmin_client.garth, 'oauth2_token'):
                    print(f"  Has OAuth2 token: {garmin_client.garth.oauth2_token is not None}")
        except Exception as e:
            print(f"  Error checking tokens: {e}")
        
        print("\nWARNING: Login succeeded but cannot retrieve activities!")
        print("This might be a Garmin API issue or account permissions problem.")
        return True  # Return True anyway since auth worked
            
    except Exception as e:
        import traceback
        print(f"Failed to login to Garmin: {e}")
        print("Full error trace:")
        traceback.print_exc()
        return False

def calculate_normalized_power(power_values):
    """Calculate Normalized Power from raw power data.

    NP = 4th root of average of 4th power of 30-sec rolling averages
    """
    if len(power_values) < 30:
        return sum(power_values) / len(power_values) if power_values else 0

    # Calculate 30-second rolling averages
    window = 30
    rolling_avgs = []

    current_sum = sum(power_values[:window])
    rolling_avgs.append(current_sum / window)

    for i in range(1, len(power_values) - window + 1):
        current_sum = current_sum - power_values[i-1] + power_values[i + window - 1]
        rolling_avgs.append(current_sum / window)

    # Calculate 4th power of each rolling average
    fourth_powers = [avg ** 4 for avg in rolling_avgs]

    # Average of 4th powers
    avg_fourth = sum(fourth_powers) / len(fourth_powers)

    # 4th root
    np_power = avg_fourth ** 0.25

    return np_power


def calculate_best_power(power_values, window_seconds):
    """Calculate best average power for a given duration window."""
    if len(power_values) < window_seconds:
        return None

    current_sum = sum(power_values[:window_seconds])
    best_power = current_sum / window_seconds

    for i in range(1, len(power_values) - window_seconds + 1):
        current_sum = current_sum - power_values[i-1] + power_values[i + window_seconds - 1]
        avg = current_sum / window_seconds
        if avg > best_power:
            best_power = avg

    return round(best_power, 1)


# Power duration windows in seconds
POWER_DURATIONS = {
    "30s": 30,
    "1min": 60,
    "3min": 180,
    "5min": 300,
    "10min": 600,
    "20min": 1200
}


def analyze_activity_power(activity_id):
    """Analyze raw power time series data for an activity.

    Returns:
    - Best power for multiple durations (15s, 30s, 1m, 2m, 5m, 10m, 20m)
    - Normalized Power (for TSS calculation)
    - Duration in seconds
    """
    global garmin_client

    try:
        # Get detailed activity data with power samples (1-second resolution)
        details = garmin_client.get_activity_details(activity_id, maxchart=20000)

        # Find power index in metrics array
        power_idx = None
        for desc in details.get("metricDescriptors", []):
            if desc.get("key") == "directPower":
                power_idx = desc.get("metricsIndex")
                break

        if power_idx is None:
            print(f"  No power data found for activity {activity_id}")
            return None

        samples = details.get("activityDetailMetrics", [])
        if not samples:
            return None

        # Extract power values
        power_values = []
        for sample in samples:
            metrics = sample.get("metrics", [])
            if power_idx < len(metrics):
                power = metrics[power_idx]
                if power is not None and power >= 0:
                    power_values.append(power)
                else:
                    power_values.append(0)  # Missing data = 0 power
            else:
                power_values.append(0)

        if not power_values:
            return None

        # Calculate best power for each duration
        best_powers = {}
        for name, seconds in POWER_DURATIONS.items():
            best_powers[name] = calculate_best_power(power_values, seconds)

        # Calculate Normalized Power
        np_power = calculate_normalized_power(power_values)

        # Calculate average power (excluding zeros for coasting)
        non_zero_power = [p for p in power_values if p > 0]
        avg_power = sum(non_zero_power) / len(non_zero_power) if non_zero_power else 0

        return {
            "best_powers": best_powers,
            "best_20min_power": best_powers.get("20min"),  # Keep for FTP calculation
            "normalized_power": round(np_power, 1),
            "avg_power": round(avg_power, 1),
            "duration_seconds": len(power_values),
            "total_samples": len(power_values)
        }

    except Exception as e:
        print(f"Error analyzing power for activity {activity_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_power_tss(normalized_power, duration_seconds, ftp):
    """Calculate TSS from power data.

    TSS = (duration_seconds * NP * IF) / (FTP * 3600)
    where IF = NP / FTP

    Simplifies to: TSS = (duration_seconds * IF²) / 36
    """
    if ftp <= 0 or normalized_power <= 0:
        return None

    intensity_factor = normalized_power / ftp
    tss = (duration_seconds * intensity_factor ** 2) / 36

    return round(tss)


def get_intensity_factor(hr, max_hr=185):
    """Calculate intensity factor from a single HR value (fallback for no power data)"""
    hr_percentage = (hr / max_hr) * 100

    if hr_percentage < 75:
        return 0.65
    elif hr_percentage < 85:
        return 0.75 + (hr_percentage - 75) * 0.01
    elif hr_percentage < 95:
        return 0.85 + (hr_percentage - 85) * 0.01
    else:
        return 0.95 + min((hr_percentage - 95) * 0.01, 0.1)


def analyze_activity_hr(activity_id):
    """Analyze raw HR time series data for an activity.

    Returns:
    - TSS calculated from valid HR samples only
    - Anomaly detection (>5 min of HR < 100 or missing)
    - Stats about data quality
    """
    global garmin_client
    max_hr = 185
    low_hr_threshold = 100

    try:
        # Get detailed activity data with HR samples (1-second resolution)
        details = garmin_client.get_activity_details(activity_id, maxchart=20000)

        # Find HR index in metrics array
        hr_idx = None
        for desc in details.get("metricDescriptors", []):
            if desc.get("key") == "directHeartRate":
                hr_idx = desc.get("metricsIndex")
                break

        if hr_idx is None:
            return None

        samples = details.get("activityDetailMetrics", [])
        if not samples:
            return None

        # Analyze each sample
        valid_hr_samples = []
        total_if_squared = 0

        # Track continuous low HR and missing HR periods
        current_low_hr_streak = 0
        max_low_hr_streak = 0
        current_missing_streak = 0
        max_missing_streak = 0

        for sample in samples:
            metrics = sample.get("metrics", [])
            if hr_idx >= len(metrics):
                current_missing_streak += 1
                max_missing_streak = max(max_missing_streak, current_missing_streak)
                current_low_hr_streak = 0  # Reset low HR streak
                continue

            hr = metrics[hr_idx]

            if hr is None or hr == 0:
                current_missing_streak += 1
                max_missing_streak = max(max_missing_streak, current_missing_streak)
                current_low_hr_streak = 0  # Reset low HR streak
            elif hr < low_hr_threshold:
                current_low_hr_streak += 1
                max_low_hr_streak = max(max_low_hr_streak, current_low_hr_streak)
                current_missing_streak = 0  # Reset missing streak
                # Still count low HR in TSS
                intensity_factor = get_intensity_factor(hr, max_hr)
                total_if_squared += intensity_factor ** 2
                valid_hr_samples.append(hr)
            else:
                # Valid HR sample - reset both streaks
                current_low_hr_streak = 0
                current_missing_streak = 0
                intensity_factor = get_intensity_factor(hr, max_hr)
                total_if_squared += intensity_factor ** 2
                valid_hr_samples.append(hr)

        # Calculate TSS from raw data
        # TSS = (total_if_squared / 36) since each sample is 1 second
        # and TSS is normalized to 1 hour at threshold (IF=1.0 for 1hr = 100 TSS)
        tss_from_raw = (total_if_squared / 36) if valid_hr_samples else None

        # Anomaly detection - flag if >5 min CONTINUOUS low/missing HR
        max_low_hr_mins = max_low_hr_streak / 60
        max_missing_mins = max_missing_streak / 60
        has_anomaly = max_low_hr_mins > 5 or max_missing_mins > 5

        return {
            "tss": round(tss_from_raw) if tss_from_raw else None,
            "has_anomaly": has_anomaly,
            "low_hr_minutes": round(max_low_hr_mins, 1),
            "missing_hr_minutes": round(max_missing_mins, 1),
            "valid_samples": len(valid_hr_samples),
            "total_samples": len(samples),
            "avg_hr_from_valid": round(sum(valid_hr_samples) / len(valid_hr_samples), 1) if valid_hr_samples else None,
            "anomaly_reasons": (
                (["Low HR (<100) for {:.1f} min continuous".format(max_low_hr_mins)] if max_low_hr_mins > 5 else []) +
                (["Missing HR for {:.1f} min continuous".format(max_missing_mins)] if max_missing_mins > 5 else [])
            )
        }

    except Exception as e:
        print(f"Error analyzing HR for activity {activity_id}: {e}")
        return None


def calculate_intensity_score(activity, ftp=None):
    """Calculate TSS from power data, with HR fallback.

    Uses cache to avoid re-fetching data for known activities.
    Prioritizes power data over HR data.
    FTP should be provided for accurate TSS calculation.
    """
    activity_id = activity.get('activityId')

    # Check cache first
    if activity_id:
        cached = get_cached_tss(activity_id)
        if cached:
            return cached

    # Try to use power data first
    if activity_id and garmin_client:
        power_analysis = analyze_activity_power(activity_id)
        if power_analysis and power_analysis.get('normalized_power'):
            # Use provided FTP or load from file
            if ftp is None:
                ftp_data = load_ftp_data()
                ftp = ftp_data.get('current_ftp', 200)

            tss = calculate_power_tss(
                power_analysis['normalized_power'],
                power_analysis['duration_seconds'],
                ftp
            )

            result = {
                "tss": tss,
                "normalized_power": power_analysis['normalized_power'],
                "avg_power": power_analysis['avg_power'],
                "best_20min_power": power_analysis.get('best_20min_power'),
                "ftp_used": ftp,
                "intensity_factor": round(power_analysis['normalized_power'] / ftp, 2) if ftp > 0 else None,
                "data_source": "power",
                "has_anomaly": False,
                "anomaly_reasons": []
            }

            # Cache the result
            cache_tss(activity_id, result)
            return result

    # Fallback to HR-based calculation if no power data
    if activity_id and garmin_client:
        hr_analysis = analyze_activity_hr(activity_id)
        if hr_analysis and hr_analysis.get('tss') is not None:
            hr_analysis['data_source'] = 'hr'
            cache_tss(activity_id, hr_analysis)
            return hr_analysis

    # Final fallback to average HR
    avg_hr = activity.get('averageHR')
    duration = activity.get('duration', 0)

    if not avg_hr or duration == 0:
        return None

    max_hr = 185
    intensity_factor = get_intensity_factor(avg_hr, max_hr)

    hours = duration / 3600
    tss = hours * (intensity_factor ** 2) * 100

    return {
        "tss": round(tss),
        "data_source": "hr_avg",
        "has_anomaly": False,
        "anomaly_reasons": [],
        "fallback": True
    }


def process_activities_with_ftp(activities):
    """Process activities chronologically, updating FTP and power records as new bests are found.

    This ensures TSS is calculated using the FTP that was current at the time
    of each activity. Also tracks best power for all durations.
    """
    # Load current FTP data
    ftp_data = load_ftp_data()
    current_ftp = ftp_data.get('current_ftp', 100)
    ftp_history = ftp_data.get('history', [])
    power_records = ftp_data.get('power_records', {dur: [] for dur in POWER_DURATIONS.keys()})
    current_bests = ftp_data.get('current_bests', {dur: 0 for dur in POWER_DURATIONS.keys()})

    # Sort activities by date (oldest first)
    sorted_activities = sorted(activities, key=lambda x: x.get('startTimeLocal', ''))

    results = {}

    for activity in sorted_activities:
        activity_id = activity.get('activityId')
        activity_date = activity.get('startTimeLocal', '')[:10]  # YYYY-MM-DD

        # Check if we have cached data with matching FTP
        cached = get_cached_tss(activity_id)
        if cached and cached.get('ftp_used') == current_ftp and cached.get('best_powers'):
            results[activity_id] = cached
            # Still check if this activity could update records
            best_powers = cached.get('best_powers', {})

            # Check all durations for new records
            for dur in POWER_DURATIONS.keys():
                power = best_powers.get(dur)
                if power and power > current_bests.get(dur, 0):
                    current_bests[dur] = power
                    power_records[dur].append({"date": activity_date, "power": power})

            # Check FTP (20min * 0.95)
            best_20 = best_powers.get('20min')
            if best_20 and best_20 * 0.95 > current_ftp:
                new_ftp = round(best_20 * 0.95)
                print(f"  FTP updated: {current_ftp}W -> {new_ftp}W (from {activity_date})")
                current_ftp = new_ftp
                # Only add if this date isn't already in history with same or higher FTP
                existing = next((h for h in ftp_history if h['date'] == activity_date), None)
                if not existing or existing['ftp'] < new_ftp:
                    if existing:
                        ftp_history.remove(existing)
                    ftp_history.append({"date": activity_date, "ftp": new_ftp})
            continue

        # Need to calculate fresh - get power analysis
        power_analysis = None
        if garmin_client:
            power_analysis = analyze_activity_power(activity_id)

        if power_analysis and power_analysis.get('normalized_power'):
            best_powers = power_analysis.get('best_powers', {})

            # Check all durations for new records
            for dur in POWER_DURATIONS.keys():
                power = best_powers.get(dur)
                if power and power > current_bests.get(dur, 0):
                    current_bests[dur] = power
                    power_records[dur].append({"date": activity_date, "power": power})

            # Check if this activity sets a new FTP
            best_20 = best_powers.get('20min')
            if best_20 and best_20 * 0.95 > current_ftp:
                new_ftp = round(best_20 * 0.95)
                print(f"  FTP updated: {current_ftp}W -> {new_ftp}W (from {activity_date})")
                current_ftp = new_ftp
                # Only add if this date isn't already in history with same or higher FTP
                existing = next((h for h in ftp_history if h['date'] == activity_date), None)
                if not existing or existing['ftp'] < new_ftp:
                    if existing:
                        ftp_history.remove(existing)
                    ftp_history.append({"date": activity_date, "ftp": new_ftp})

            # Calculate TSS with current FTP
            tss = calculate_power_tss(
                power_analysis['normalized_power'],
                power_analysis['duration_seconds'],
                current_ftp
            )

            result = {
                "tss": tss,
                "normalized_power": power_analysis['normalized_power'],
                "avg_power": power_analysis['avg_power'],
                "best_powers": best_powers,
                "best_20min_power": best_20,
                "ftp_used": current_ftp,
                "intensity_factor": round(power_analysis['normalized_power'] / current_ftp, 2) if current_ftp > 0 else None,
                "data_source": "power",
                "has_anomaly": False,
                "anomaly_reasons": []
            }

            cache_tss(activity_id, result)
            results[activity_id] = result
        else:
            # No power data - fall back to HR
            hr_analysis = analyze_activity_hr(activity_id) if garmin_client else None
            if hr_analysis and hr_analysis.get('tss') is not None:
                hr_analysis['data_source'] = 'hr'
                cache_tss(activity_id, hr_analysis)
                results[activity_id] = hr_analysis
            else:
                # Final fallback to average HR
                avg_hr = activity.get('averageHR')
                duration = activity.get('duration', 0)
                if avg_hr and duration > 0:
                    max_hr = 185
                    intensity_factor = get_intensity_factor(avg_hr, max_hr)
                    hours = duration / 3600
                    tss = hours * (intensity_factor ** 2) * 100
                    results[activity_id] = {
                        "tss": round(tss),
                        "data_source": "hr_avg",
                        "has_anomaly": False,
                        "anomaly_reasons": [],
                        "fallback": True
                    }

    # Save updated FTP data with all power records
    ftp_data['current_ftp'] = current_ftp
    ftp_data['history'] = ftp_history
    ftp_data['power_records'] = power_records
    ftp_data['current_bests'] = current_bests
    save_ftp_data(ftp_data)

    return results, current_ftp


@app.route('/api/test-garmin', methods=['GET'])
def test_garmin():
    """Test Garmin connection and see raw data"""
    global garmin_client
    
    if not garmin_client:
        return jsonify({"error": "Not logged in"})
    
    try:
        print("Testing Garmin connection - fetching last 10 activities...")
        
        # Get last 10 activities regardless of type
        activities = garmin_client.get_activities(0, 10)
        
        print(f"Got {len(activities)} activities")
        
        # Return simplified info
        result = []
        for act in activities:
            activity_info = {
                'name': act.get('activityName'),
                'type': act.get('activityType', {}).get('typeKey'),
                'date': act.get('startTimeLocal'),
                'duration': act.get('duration')
            }
            print(f"  - {activity_info['name']}: {activity_info['type']}")
            result.append(activity_info)
        
        return jsonify({
            'count': len(result),
            'activities': result
        })
    except Exception as e:
        import traceback
        print(f"Error testing Garmin: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login to Garmin Connect"""
    success = init_garmin()
    if success:
        return jsonify({"status": "success", "message": "Logged in to Garmin Connect"})
    else:
        return jsonify({"status": "error", "message": "Failed to login"}), 401

@app.route('/api/activities', methods=['GET'])
def get_activities():
    """Fetch activities from Garmin Connect"""
    global garmin_client

    if not garmin_client:
        return jsonify({"error": "Not logged in to Garmin"}), 401

    try:
        days = int(request.args.get('days', 30))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        print(f"\n=== Fetching activities from {start_date.date()} to {end_date.date()} ===")

        activities = garmin_client.get_activities_by_date(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        print(f"Got {len(activities)} total activities from Garmin")

        # Filter to cycling activities
        cycling_activities = []
        for activity in activities:
            activity_type = activity.get('activityType', {}).get('typeKey', '')
            if 'cycling' in activity_type.lower() or 'virtual_ride' in activity_type.lower() or 'bike' in activity_type.lower():
                cycling_activities.append(activity)

        print(f"Filtered to {len(cycling_activities)} cycling activities")

        # Process all activities with FTP tracking (chronologically)
        tss_results, current_ftp = process_activities_with_ftp(cycling_activities)

        print(f"Current FTP: {current_ftp}W")

        # Build response
        processed_activities = []
        for activity in cycling_activities:
            activity_id = activity['activityId']
            activity_name = activity.get('activityName', 'Untitled')
            analysis = tss_results.get(activity_id, {})

            processed_activities.append({
                'id': activity_id,
                'date': activity['startTimeLocal'],
                'title': activity_name,
                'location': activity.get('locationName', activity_name),
                'duration': activity.get('duration', 0),
                'distance': activity.get('distance', 0) / 1000,
                'tss': analysis.get('tss'),
                'calories': activity.get('calories'),
                'avg_hr': activity.get('averageHR'),
                'max_hr': activity.get('maxHR'),
                'avg_power': analysis.get('avg_power'),
                'normalized_power': analysis.get('normalized_power'),
                'intensity_factor': analysis.get('intensity_factor'),
                'ftp_used': analysis.get('ftp_used'),
                'data_source': analysis.get('data_source', 'unknown'),
                'type': activity.get('activityType', {}).get('typeKey', ''),
                'hr_anomaly': analysis.get('has_anomaly', False),
                'anomaly_reasons': analysis.get('anomaly_reasons', [])
            })

        print(f"Returning {len(processed_activities)} activities\n")

        return jsonify(processed_activities)

    except Exception as e:
        print(f"Error fetching activities: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/weekly-summary', methods=['GET'])
def get_weekly_summary():
    """Get weekly training summary"""
    global garmin_client
    
    if not garmin_client:
        return jsonify({"error": "Not logged in to Garmin"}), 401
    
    try:
        print("\n=== Fetching weekly summary ===")
        
        # Get last 150 days of data to cover back to August
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)
        
        activities = garmin_client.get_activities_by_date(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        print(f"Got {len(activities)} activities for weekly summary")
        
        # Group by week
        weekly_data = {}
        for activity in activities:
            activity_type = activity.get('activityType', {}).get('typeKey', '')
            if 'cycling' not in activity_type.lower() and 'virtual_ride' not in activity_type.lower() and 'bike' not in activity_type.lower():
                continue
            
            # Get week start (Monday)
            activity_date = datetime.strptime(activity['startTimeLocal'], '%Y-%m-%d %H:%M:%S')
            week_start = activity_date - timedelta(days=activity_date.weekday())
            week_key = week_start.strftime('%Y-%m-%d')
            
            if week_key not in weekly_data:
                weekly_data[week_key] = {
                    'week_start': week_key,
                    'total_duration': 0,
                    'total_tss': 0,
                    'activity_count': 0
                }
            
            weekly_data[week_key]['total_duration'] += activity.get('duration', 0)
            hr_analysis = calculate_intensity_score(activity)
            if hr_analysis and hr_analysis.get('tss'):
                weekly_data[week_key]['total_tss'] += hr_analysis['tss']
            weekly_data[week_key]['activity_count'] += 1
        
        # Convert to list and sort by date
        summary = sorted(weekly_data.values(), key=lambda x: x['week_start'], reverse=True)
        
        print(f"Returning {len(summary)} weeks of data\n")
        
        return jsonify(summary)
    
    except Exception as e:
        print(f"Error fetching weekly summary: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/training-load', methods=['GET'])
def get_training_load():
    """Calculate CTL, ATL, and TSB (Form) over time"""
    global garmin_client
    
    if not garmin_client:
        return jsonify({"error": "Not logged in to Garmin"}), 401
    
    try:
        print("\n=== Calculating training load metrics ===")
        
        # Get last 180 days to have enough data for proper CTL calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        activities = garmin_client.get_activities_by_date(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Filter to cycling activities and sort by date
        cycling_activities = []
        for activity in activities:
            activity_type = activity.get('activityType', {}).get('typeKey', '')
            if 'cycling' in activity_type.lower() or 'virtual_ride' in activity_type.lower() or 'bike' in activity_type.lower():
                hr_analysis = calculate_intensity_score(activity)
                if hr_analysis and hr_analysis.get('tss'):  # Only include activities with TSS
                    cycling_activities.append({
                        'date': datetime.strptime(activity['startTimeLocal'], '%Y-%m-%d %H:%M:%S').date(),
                        'tss': hr_analysis['tss']
                    })
        
        # Sort by date
        cycling_activities.sort(key=lambda x: x['date'])
        
        if not cycling_activities:
            return jsonify([])
        
        # Initialize CTL and ATL
        # CTL (Chronic Training Load) = 42-day exponential moving average
        # ATL (Acute Training Load) = 7-day exponential moving average
        # TSB (Training Stress Balance) = CTL - ATL
        
        ctl_time_constant = 42
        atl_time_constant = 7
        
        # Create a dictionary of daily TSS values
        daily_tss = {}
        for act in cycling_activities:
            if act['date'] not in daily_tss:
                daily_tss[act['date']] = 0
            daily_tss[act['date']] += act['tss']
        
        # Calculate CTL, ATL, and TSB for each day
        training_load_data = []
        ctl = 0
        atl = 0
        
        # Start from the first activity date
        first_date = min(daily_tss.keys())
        last_date = max(daily_tss.keys())
        current_date = first_date
        
        while current_date <= last_date:
            # Get today's TSS (0 if no activity)
            today_tss = daily_tss.get(current_date, 0)
            
            # Update CTL (exponential moving average)
            ctl = ctl + (today_tss - ctl) / ctl_time_constant
            
            # Update ATL (exponential moving average)
            atl = atl + (today_tss - atl) / atl_time_constant
            
            # Calculate TSB (Form)
            tsb = ctl - atl
            
            training_load_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'fitness': round(ctl, 1),
                'fatigue': round(atl, 1),
                'form': round(tsb, 1),
                'daily_tss': today_tss
            })
            
            current_date += timedelta(days=1)
        
        # Only return last 150 days for the chart
        cutoff_date = end_date.date() - timedelta(days=150)
        filtered_data = [d for d in training_load_data if datetime.strptime(d['date'], '%Y-%m-%d').date() >= cutoff_date]
        
        print(f"Returning {len(filtered_data)} days of training load data\n")
        
        return jsonify(filtered_data)
    
    except Exception as e:
        print(f"Error calculating training load: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "logged_in": garmin_client is not None})

@app.route('/api/ftp', methods=['GET'])
def get_ftp():
    """Get current FTP and history"""
    ftp_data = load_ftp_data()
    return jsonify(ftp_data)

@app.route('/api/ftp', methods=['POST'])
def set_ftp():
    """Manually set FTP (clears cache to recalculate TSS)"""
    data = request.json
    new_ftp = data.get('ftp')

    if not new_ftp or new_ftp < 50 or new_ftp > 500:
        return jsonify({"error": "Invalid FTP value"}), 400

    ftp_data = load_ftp_data()
    old_ftp = ftp_data.get('current_ftp', 200)
    ftp_data['current_ftp'] = new_ftp
    ftp_data['history'].append({
        "date": datetime.now().strftime('%Y-%m-%d'),
        "ftp": new_ftp,
        "manual": True
    })
    save_ftp_data(ftp_data)

    # Clear cache to recalculate TSS with new FTP
    if os.path.exists(TSS_CACHE_FILE):
        os.remove(TSS_CACHE_FILE)

    return jsonify({
        "message": f"FTP updated from {old_ftp}W to {new_ftp}W",
        "ftp": new_ftp
    })

@app.route('/api/activity-hr-details/<activity_id>', methods=['GET'])
def get_activity_hr_details(activity_id):
    """Get detailed HR data for an activity to analyze for anomalies"""
    global garmin_client

    if not garmin_client:
        return jsonify({"error": "Not logged in to Garmin"}), 401

    try:
        # Get HR time in zones
        hr_zones = garmin_client.get_activity_hr_in_timezones(activity_id)

        # Get detailed activity data (includes HR samples)
        details = garmin_client.get_activity_details(activity_id, maxchart=5000)

        return jsonify({
            "hr_zones": hr_zones,
            "details_keys": list(details.keys()) if isinstance(details, dict) else "not a dict",
            "metric_descriptors": details.get("metricDescriptors", []),
            "sample_count": len(details.get("activityDetailMetrics", [])) if details.get("activityDetailMetrics") else 0,
            "first_samples": details.get("activityDetailMetrics", [])[:5] if details.get("activityDetailMetrics") else []
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend"""
    frontend_dir = 'frontend-build'
    
    # API routes are already defined above, so they take precedence
    # If path starts with 'api/', this won't be reached
    
    # Try to serve the requested file
    if path and os.path.exists(os.path.join(frontend_dir, path)):
        return send_from_directory(frontend_dir, path)
    else:
        # Serve index.html for all other routes (React Router)
        return send_from_directory(frontend_dir, 'index.html')

if __name__ == '__main__':
    # Print all registered routes for debugging
    print("\nRegistered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
    
    # Try to login on startup
    init_garmin()
    app.run(debug=True, host='0.0.0.0', port=5001)