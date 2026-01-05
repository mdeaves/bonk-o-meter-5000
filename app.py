from flask import Flask, jsonify, request
from flask_cors import CORS
from garminconnect import Garmin
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}})

# Garmin client instance
garmin_client = None

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

def calculate_intensity_score(activity):
    """Calculate intensity score from heart rate and duration"""
    # Try to get TSS from Garmin first
    if 'trainingStressScore' in activity:
        return activity['trainingStressScore']
    
    # Calculate from heart rate if available
    avg_hr = activity.get('averageHR')
    duration = activity.get('duration', 0)
    
    # Return None if no heart rate data - we want consistent calculation method
    if not avg_hr or duration == 0:
        return None
    
    # Use actual max HR of 185
    max_hr = 185
    
    # Calculate intensity factor (IF) as a percentage of max HR
    # Zone 1-2 (< 75% max HR): IF ~0.65
    # Zone 3 (75-85% max HR): IF ~0.75-0.85
    # Zone 4 (85-95% max HR): IF ~0.85-0.95
    # Zone 5 (> 95% max HR): IF ~0.95-1.05
    
    hr_percentage = (avg_hr / max_hr) * 100
    
    if hr_percentage < 75:
        intensity_factor = 0.65
    elif hr_percentage < 85:
        intensity_factor = 0.75 + (hr_percentage - 75) * 0.01
    elif hr_percentage < 95:
        intensity_factor = 0.85 + (hr_percentage - 85) * 0.01
    else:
        intensity_factor = 0.95 + min((hr_percentage - 95) * 0.01, 0.1)
    
    # TSS = (duration in hours) × (IF^2) × 100
    hours = duration / 3600
    tss = hours * (intensity_factor ** 2) * 100
    
    return round(tss)

@app.route('/')
def home():
    return "Bonk-O-Meter 5000 Backend is running!"

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
        
        # Process activities into our format
        processed_activities = []
        cycling_count = 0
        
        for activity in activities:
            activity_type = activity.get('activityType', {}).get('typeKey', '')
            activity_name = activity.get('activityName', 'Untitled')
            
            print(f"  Activity: '{activity_name}' - Type: '{activity_type}'")
            
            if 'cycling' in activity_type.lower() or 'virtual_ride' in activity_type.lower() or 'bike' in activity_type.lower():
                cycling_count += 1
                
                # Calculate intensity score
                intensity_score = calculate_intensity_score(activity)
                
                processed_activities.append({
                    'id': activity['activityId'],
                    'date': activity['startTimeLocal'],
                    'title': activity_name,
                    'location': activity.get('locationName', activity_name),
                    'duration': activity.get('duration', 0),
                    'distance': activity.get('distance', 0) / 1000,
                    'tss': intensity_score,
                    'calories': activity.get('calories'),
                    'avg_hr': activity.get('averageHR'),
                    'max_hr': activity.get('maxHR'),
                    'type': activity_type
                })
        
        print(f"Filtered to {cycling_count} cycling activities")
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
            intensity_score = calculate_intensity_score(activity)
            if intensity_score:
                weekly_data[week_key]['total_tss'] += intensity_score
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
                intensity_score = calculate_intensity_score(activity)
                if intensity_score:  # Only include activities with TSS
                    cycling_activities.append({
                        'date': datetime.strptime(activity['startTimeLocal'], '%Y-%m-%d %H:%M:%S').date(),
                        'tss': intensity_score
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

if __name__ == '__main__':
    # Print all registered routes for debugging
    print("\nRegistered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
    
    # Try to login on startup
    init_garmin()
    app.run(debug=True, port=5000)