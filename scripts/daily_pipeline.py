# scripts/daily_pipeline.py
"""
Complete daily horse racing prediction pipeline.

Workflow:
1. Update historical data via rpscrape and copy to training directory
2. Re-encode training data with latest history
3. Process yesterday's results in Google Sheets
4. Download today's racecard
5. Generate cleansed predictions
6. Update Google Sheets with new predictions
7. Optional: Send notifications

Usage:
  python daily_pipeline.py                    # Interactive mode (default)
  python daily_pipeline.py --run-all          # Non-interactive: run all remaining steps
  python daily_pipeline.py --status           # Show current status and exit
  python daily_pipeline.py --reset            # Reset all statuses and exit
"""

import sys
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import time
import pickle
import argparse

from common import DATA_DIR, PROJECT_DIR, convert_to_24h_time
from google_sheets_writer import setup_google_sheets, write_formatted_predictions_to_sheet
from sheets_manager import SheetsManager

# Status tracking
STATUS_FILE = PROJECT_DIR / 'logs' / f'pipeline_status_{datetime.now().strftime("%Y-%m-%d")}.pkl'
STATUS_FILE.parent.mkdir(exist_ok=True)

class PipelineStatus:
    """Track pipeline step completion status with dependency management"""
    
    def __init__(self):
        self.steps = {
            "1": {"name": "Update Historical Data", "completed": False, "timestamp": None, "success": None, "depends_on": []},
            "2": {"name": "Re-encode Training Data", "completed": False, "timestamp": None, "success": None, "depends_on": ["1"]},
            "3": {"name": "Process Yesterday's Results", "completed": False, "timestamp": None, "success": None, "depends_on": []},
            "4": {"name": "Download Today's Racecard", "completed": False, "timestamp": None, "success": None, "depends_on": []},
            "5": {"name": "Generate Predictions", "completed": False, "timestamp": None, "success": None, "depends_on": ["2", "4"]},
            "6": {"name": "Update Google Sheets", "completed": False, "timestamp": None, "success": None, "depends_on": ["5"]},
            "7": {"name": "Send Notifications", "completed": False, "timestamp": None, "success": None, "depends_on": ["6"]},
        }
        self.load_status()
    
    def load_status(self):
        """Load existing status from file"""
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE, 'rb') as f:
                    saved_status = pickle.load(f)
                    # Update steps but preserve dependency info
                    for step_num, step_data in saved_status.items():
                        if step_num in self.steps:
                            self.steps[step_num].update({
                                "completed": step_data.get("completed", False),
                                "timestamp": step_data.get("timestamp", None),
                                "success": step_data.get("success", None)
                            })
                print(f"📂 Loaded pipeline status from {STATUS_FILE}")
            except Exception as e:
                print(f"⚠️  Could not load status file: {e}")
    
    def save_status(self):
        """Save current status to file"""
        try:
            with open(STATUS_FILE, 'wb') as f:
                pickle.dump(self.steps, f)
        except Exception as e:
            print(f"⚠️  Could not save status: {e}")
    
    def mark_completed(self, step_num, success=True):
        """Mark a step as completed and invalidate dependent steps"""
        if step_num in self.steps:
            self.steps[step_num]["completed"] = True
            self.steps[step_num]["success"] = success
            self.steps[step_num]["timestamp"] = datetime.now().strftime("%H:%M:%S")
            
            # If this step was re-run, invalidate all dependent steps
            if success:
                self._invalidate_dependent_steps(step_num)
            
            self.save_status()
    
    def _invalidate_dependent_steps(self, step_num):
        """Invalidate steps that depend on the given step"""
        for other_step, step_info in self.steps.items():
            if step_num in step_info.get("depends_on", []):
                if step_info["completed"]:
                    print(f"⚠️  Invalidating Step {other_step} ({step_info['name']}) - depends on re-run Step {step_num}")
                    step_info["completed"] = False
                    step_info["success"] = None
                    step_info["timestamp"] = None
                    # Recursively invalidate steps that depend on this one
                    self._invalidate_dependent_steps(other_step)
    
    def can_run_step(self, step_num):
        """Check if a step can be run (all dependencies satisfied)"""
        if step_num not in self.steps:
            return False, "Invalid step number"
        
        depends_on = self.steps[step_num].get("depends_on", [])
        
        for dep_step in depends_on:
            if not self.is_completed(dep_step):
                dep_name = self.steps[dep_step]["name"]
                return False, f"Depends on Step {dep_step} ({dep_name}) which is not completed"
        
        return True, "OK"
    
    def get_next_runnable_step(self):
        """Get the next step that can be run (dependencies satisfied)"""
        for step_num in sorted(self.steps.keys()):
            if not self.is_completed(step_num):
                can_run, reason = self.can_run_step(step_num)
                if can_run:
                    return step_num
        return None
    
    def is_completed(self, step_num):
        """Check if a step is completed successfully"""
        return self.steps.get(step_num, {}).get("completed", False) and \
               self.steps.get(step_num, {}).get("success", False)
    
    def get_status_summary(self):
        """Get a summary of all step statuses"""
        summary = []
        for step_num, info in self.steps.items():
            status = "✅ DONE" if info["completed"] and info["success"] else \
                    "❌ FAILED" if info["completed"] and not info["success"] else \
                    "⏳ PENDING"
            timestamp = f" ({info['timestamp']})" if info["timestamp"] else ""
            
            # Show dependencies
            depends = info.get("depends_on", [])
            dep_str = f" [depends: {','.join(depends)}]" if depends else ""
            
            summary.append(f"Step {step_num}: {info['name']:30} | {status}{timestamp}{dep_str}")
        return summary
    
    def reset_all(self):
        """Reset all step statuses"""
        for step_num in self.steps:
            self.steps[step_num]["completed"] = False
            self.steps[step_num]["success"] = None
            self.steps[step_num]["timestamp"] = None
        self.save_status()
        print("🔄 All step statuses reset")
    
    def get_next_step(self):
        """Get the next step that needs to be run (for backward compatibility)"""
        return self.get_next_runnable_step()

def run_command(command, description, working_dir=None):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        if working_dir:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True, cwd=working_dir)
        else:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
        
        if result.stdout:
            print(f"✅ Success: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def update_historical_data():
    """Step 1: Update historical data via rpscrape and copy to training directory"""
    print("\n" + "="*60)
    print("📅 STEP 1: UPDATING HISTORICAL DATA")
    print("="*60)
    
    # Get current year
    current_year = datetime.now().year
    
    # Path to rpscrape scripts directory using relative path
    rpscrape_scripts_path = PROJECT_DIR.parent.parent / 'joenano' / 'rpscrape' / 'scripts'
    
    if not rpscrape_scripts_path.exists():
        print(f"❌ RPScrape scripts directory not found at: {rpscrape_scripts_path}")
        print("💡 Please ensure rpscrape is installed at the expected location")
        return False
    
    # Get the full Python path for the current environment
    python_exe = PROJECT_DIR / '.venv' / 'Scripts' / 'python.exe'
    if not python_exe.exists():
        # Fallback to system python
        python_cmd = "python"
    else:
        python_cmd = str(python_exe)
    
    # Update current year's flat racing data
    print(f"🏇 Updating {current_year} flat racing data...")
    flat_command = f'"{python_cmd}" rpscrape.py -r gb -y {current_year} -t flat'
    flat_success = run_command(flat_command, f"Downloading {current_year} GB flat racing data", rpscrape_scripts_path)
    
    if not flat_success:
        print(f"❌ Failed to update flat racing data for {current_year}")
        return False
    
    # Update current year's jumps racing data
    print(f"🐎 Updating {current_year} jumps racing data...")
    jumps_command = f'"{python_cmd}" rpscrape.py -r gb -y {current_year} -t jumps'
    jumps_success = run_command(jumps_command, f"Downloading {current_year} GB jumps racing data", rpscrape_scripts_path)
    
    if not jumps_success:
        print(f"❌ Failed to update jumps racing data for {current_year}")
        return False
    
    # Copy the downloaded data to our training data directory
    print(f"📂 Copying downloaded data to training directory...")
    copy_command = f'"{python_cmd}" scripts/copy_raw_data.py'
    copy_success = run_command(copy_command, "Copying rpscrape data to training directory", PROJECT_DIR)
    
    if not copy_success:
        print(f"❌ Failed to copy data to training directory")
        return False
    
    print(f"✅ Historical data updated for {current_year} (both flat and jumps)")
    return True

def re_encode_training_data():
    """Step 2: Re-encode training data with latest historical data"""
    print("\n" + "="*60)
    print("🔧 STEP 2: RE-ENCODING TRAINING DATA")
    print("="*60)
    
    command = "python scripts/encode_data.py"
    success = run_command(command, "Re-encoding training data with latest history")
    
    if success:
        print("✅ Training data re-encoded successfully")
    
    return success

def process_yesterday_results():
    """Step 3: Process yesterday's results in Google Sheets"""
    print("\n" + "="*60)
    print("📊 STEP 3: PROCESSING YESTERDAY'S RESULTS")
    print("="*60)
    
    try:
        sheets_manager = SheetsManager()
        if not sheets_manager.client:
            print("❌ Google Sheets not available - skipping results processing")
            return False
        
        # Get yesterday's date
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Check for yesterday's data in active sheet
        yesterday_data = sheets_manager.get_yesterday_results()
        
        if yesterday_data is not None and len(yesterday_data) > 0:
            print(f"📋 Found {len(yesterday_data)} predictions for {yesterday}")
            print("💡 Please manually fill in results (Final Odds, Result, Position, P&L) in the Active sheet")
            print("💡 Then run the move-to-historical function when ready")
            
            # Option to move to historical now (if results are already filled)
            try:
                # Check if results are filled (non-empty Result column)
                if 'Result' in yesterday_data.columns:
                    filled_results = yesterday_data[yesterday_data['Result'].str.strip() != '']
                    if len(filled_results) > 0:
                        print(f"✅ Found {len(filled_results)} results already filled")
                        move_now = input("Move filled results to historical? (y/n): ").lower().strip()
                        if move_now in ['y', 'yes']:
                            success = sheets_manager.move_to_historical(yesterday)
                            if success:
                                print("✅ Results moved to historical sheet")
                            return success
            except:
                pass  # Continue if no interactive input available
            
        else:
            print(f"📋 No predictions found for {yesterday} in active sheet")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing yesterday's results: {e}")
        return False

def download_todays_racecard():
    """Step 4: Download today's racecard"""
    print("\n" + "="*60)
    print("🏇 STEP 4: DOWNLOADING TODAY'S RACECARD")
    print("="*60)
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Check if racecard already exists
    racecard_file = DATA_DIR / 'prediction' / 'raw' / f"{today}.json"
    
    if racecard_file.exists():
        print(f"✅ Today's racecard already exists: {racecard_file}")
        return True
    
    # TODO: Implement rpscrape racecard download
    # This would involve calling rpscrape with today's date
    rpscrape_path = PROJECT_DIR.parent.parent / 'joenano' / 'rpscrape'
    
    if rpscrape_path.exists():
        # Download today's racecard
        command = f"python rpscrape.py --racecard {today}"
        success = run_command(command, f"Downloading today's racecard ({today})", rpscrape_path)
        
        if success:
            print(f"✅ Today's racecard downloaded")
            return True
    
    print("⚠️  RPScrape racecard download not implemented")
    print("💡 Please manually download today's racecard or implement rpscrape integration")
    return False

def generate_predictions():
    """Step 5: Generate cleansed predictions"""
    print("\n" + "="*60)
    print("🎯 STEP 5: GENERATING PREDICTIONS")
    print("="*60)
    
    # Run cleansing script
    print("🧹 Cleansing today's racecard...")
    success = run_command("python scripts/cleanse_racecard.py", "Cleansing racecard data")
    
    if not success:
        print("❌ Failed to cleanse racecard")
        return False
    
    # Generate predictions
    print("🤖 Generating predictions...")
    success = run_command("python scripts/predict.py", "Generating predictions")
    
    if success:
        print("✅ Predictions generated successfully")
    
    return success

def update_google_sheets():
    """Step 6: Update Google Sheets with new predictions"""
    print("\n" + "="*60)
    print("📈 STEP 6: UPDATING GOOGLE SHEETS")
    print("="*60)
    
    try:
        sheets_manager = SheetsManager()
        if not sheets_manager.client:
            print("❌ Google Sheets not available - skipping sheet update")
            return False
        
        # Load today's predictions
        today = datetime.now().strftime('%Y-%m-%d')
        predictions_file = DATA_DIR / f"predictions_{today}_calibrated.csv"
        
        if not predictions_file.exists():
            print(f"❌ No predictions file found: {predictions_file}")
            return False
        
        df = pd.read_csv(predictions_file)
        
        # Filter for significant horses
        MIN_THRESHOLD = 0.15
        significant_horses = df[df['win_probability'] >= MIN_THRESHOLD]
        
        if len(significant_horses) == 0:
            # Take top horse per race if none above threshold
            significant_horses = df.loc[df.groupby('race_id')['win_probability'].idxmax()]
            print(f"⚠️  No horses above {MIN_THRESHOLD:.0%} threshold, using top per race")
        
        # Add time_24h if not present
        if 'time_24h' not in significant_horses.columns:
            significant_horses['time_24h'] = significant_horses['time'].apply(convert_to_24h_time)
        
        # Sort predictions chronologically
        significant_horses = significant_horses.sort_values(['time_24h', 'course', 'win_probability'], 
                                                           ascending=[True, True, False])
        
        # Add predictions to active sheet
        success = sheets_manager.add_predictions_to_active(significant_horses)
        
        if success:
            print(f"✅ Added {len(significant_horses)} predictions to Active sheet")
            print("💡 You can now manually add Early Odds throughout the day")
        
        return success
        
    except Exception as e:
        print(f"❌ Error updating Google Sheets: {e}")
        return False

def send_notifications():
    """Step 7: Send notifications (optional)"""
    print("\n" + "="*60)
    print("📱 STEP 7: SENDING NOTIFICATIONS (OPTIONAL)")
    print("="*60)
    
    print("⚠️  Notification system not yet implemented")
    print("💡 Future features:")
    print("   • WhatsApp integration")
    print("   • Email notifications")
    print("   • Telegram bot")
    print("   • SMS alerts")
    
    return True

def main():
    """Run the complete daily pipeline with interactive options"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='UK Horse Racing Daily Pipeline')
    parser.add_argument('--run-all', action='store_true', 
                       help='Run all remaining steps non-interactively')
    parser.add_argument('--status', action='store_true',
                       help='Show current status and exit')
    parser.add_argument('--reset', action='store_true',
                       help='Reset all statuses and exit')
    
    args = parser.parse_args()
    
    print("🏇 UK HORSE RACING DAILY PIPELINE")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize status tracking
    status = PipelineStatus()
    
    # Handle command line modes
    if args.status:
        print("\n📊 CURRENT PIPELINE STATUS:")
        for line in status.get_status_summary():
            print(line)
        return True
    
    if args.reset:
        print("\n🔄 RESETTING ALL STEP STATUSES...")
        status.reset_all()
        print("✅ All statuses reset")
        return True
    
    if args.run_all:
        print("\n🚀 NON-INTERACTIVE MODE: Running all remaining steps...")
        success = run_all_steps(status)
        return success
    
    # Interactive mode
    # Show current status
    print("\n📊 CURRENT PIPELINE STATUS:")
    for line in status.get_status_summary():
        print(line)
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print("🎯 PIPELINE OPTIONS:")
        print("1. Run all remaining steps")
        print("2. Run specific step")
        print("3. Run from next pending step")
        print("4. Show current status")
        print("5. Reset all step statuses")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            run_all_steps(status)
        
        elif choice == "2":
            run_specific_step(status)
        
        elif choice == "3":
            run_from_next_step(status)
        
        elif choice == "4":
            print("\n📊 CURRENT STATUS:")
            for line in status.get_status_summary():
                print(line)
        
        elif choice == "5":
            confirm = input("Are you sure you want to reset all statuses? (y/n): ").lower()
            if confirm in ['y', 'yes']:
                status.reset_all()
        
        elif choice == "6":
            print("\n👋 Goodbye!")
            return True
        
        else:
            print("❌ Invalid choice. Please enter 1-6.")
    
    return True

def run_all_steps(status):
    """Run all pipeline steps, skipping completed ones"""
    print("\n🚀 RUNNING ALL REMAINING STEPS")
    print("=" * 40)
    
    # Pipeline steps
    steps = [
        ("1", "Update Historical Data", update_historical_data),
        ("2", "Re-encode Training Data", re_encode_training_data),
        ("3", "Process Yesterday's Results", process_yesterday_results),
        ("4", "Download Today's Racecard", download_todays_racecard),
        ("5", "Generate Predictions", generate_predictions),
        ("6", "Update Google Sheets", update_google_sheets),
        ("7", "Send Notifications", send_notifications),
    ]
    
    overall_success = True
    
    for step_num, step_name, step_func in steps:
        if status.is_completed(step_num):
            print(f"\n⏭️  Skipping Step {step_num}: {step_name} (already completed)")
            continue
        
        print(f"\n🔄 Running Step {step_num}: {step_name}")
        try:
            success = step_func()
            status.mark_completed(step_num, success)
            
            if not success:
                print(f"❌ Step {step_num} failed. Stopping pipeline.")
                overall_success = False
                break
                
        except Exception as e:
            print(f"❌ Unexpected error in {step_name}: {e}")
            status.mark_completed(step_num, False)
            overall_success = False
            break
        
        # Add delay between steps
        time.sleep(1)
    
    # Final summary
    print("\n" + "="*60)
    print("📊 PIPELINE SUMMARY")
    print("="*60)
    for line in status.get_status_summary():
        print(line)
    
    return overall_success

def run_specific_step(status):
    """Run a specific step"""
    print("\n🎯 SELECT STEP TO RUN:")
    steps = [
        ("1", "Update Historical Data", update_historical_data),
        ("2", "Re-encode Training Data", re_encode_training_data),
        ("3", "Process Yesterday's Results", process_yesterday_results),
        ("4", "Download Today's Racecard", download_todays_racecard),
        ("5", "Generate Predictions", generate_predictions),
        ("6", "Update Google Sheets", update_google_sheets),
        ("7", "Send Notifications", send_notifications),
    ]
    
    for step_num, step_name, _ in steps:
        completed = "✅" if status.is_completed(step_num) else "⏳"
        can_run, reason = status.can_run_step(step_num)
        runnable = "🟢" if can_run else "🔴"
        print(f"{step_num}. {step_name:30} {completed} {runnable}")
        if not can_run and not status.is_completed(step_num):
            print(f"   └─ Cannot run: {reason}")
    
    print("\n🟢 = Can run now | 🔴 = Dependencies not met")
    choice = input("\nEnter step number (1-7): ").strip()
    
    if choice in [str(i) for i in range(1, 8)]:
        step_num, step_name, step_func = steps[int(choice) - 1]
        
        # Check if step can be run
        can_run, reason = status.can_run_step(step_num)
        if not can_run:
            print(f"❌ Cannot run Step {step_num}: {reason}")
            force_run = input("Force run anyway? (y/n): ").lower()
            if force_run not in ['y', 'yes']:
                return
            print("⚠️  Running step despite unmet dependencies - this may cause issues!")
        
        if status.is_completed(step_num):
            rerun = input(f"Step {step_num} already completed. Re-run anyway? (y/n): ").lower()
            if rerun not in ['y', 'yes']:
                return
            print("⚠️  Re-running step will invalidate dependent steps!")
        
        print(f"\n🔄 Running Step {step_num}: {step_name}")
        try:
            success = step_func()
            status.mark_completed(step_num, success)
            
            if success:
                print(f"✅ Step {step_num} completed successfully")
            else:
                print(f"❌ Step {step_num} failed")
                
        except Exception as e:
            print(f"❌ Unexpected error in {step_name}: {e}")
            status.mark_completed(step_num, False)
    else:
        print("❌ Invalid step number")

def run_from_next_step(status):
    """Run from the next runnable step"""
    next_step = status.get_next_runnable_step()
    
    if next_step is None:
        print("\n✅ All steps are already completed!")
        return
    
    steps = [
        ("1", "Update Historical Data", update_historical_data),
        ("2", "Re-encode Training Data", re_encode_training_data),
        ("3", "Process Yesterday's Results", process_yesterday_results),
        ("4", "Download Today's Racecard", download_todays_racecard),
        ("5", "Generate Predictions", generate_predictions),
        ("6", "Update Google Sheets", update_google_sheets),
        ("7", "Send Notifications", send_notifications),
    ]
    
    step_name = status.steps[next_step]["name"]
    print(f"\n🔄 Next runnable step is: Step {next_step} - {step_name}")
    
    # Show what depends on this step
    depends_on = status.steps[next_step].get("depends_on", [])
    if depends_on:
        dep_names = [f"Step {dep} ({status.steps[dep]['name']})" for dep in depends_on]
        print(f"   Dependencies: {', '.join(dep_names)}")
    
    start_from = input(f"Start from Step {next_step}? (y/n): ").lower()
    
    if start_from in ['y', 'yes']:
        # Run from next step onwards, but only steps that can be run
        remaining_steps = [(num, name, func) for num, name, func in steps 
                          if int(num) >= int(next_step)]
        
        for step_num, step_name, step_func in remaining_steps:
            if status.is_completed(step_num):
                print(f"\n⏭️  Skipping Step {step_num}: {step_name} (already completed)")
                continue
            
            # Check if step can be run
            can_run, reason = status.can_run_step(step_num)
            if not can_run:
                print(f"\n🔴 Cannot run Step {step_num}: {step_name}")
                print(f"   Reason: {reason}")
                print("   Stopping pipeline - fix dependencies first")
                break
            
            print(f"\n🔄 Running Step {step_num}: {step_name}")
            try:
                success = step_func()
                status.mark_completed(step_num, success)
                
                if not success:
                    print(f"❌ Step {step_num} failed. Stopping pipeline.")
                    break
                    
            except Exception as e:
                print(f"❌ Unexpected error in {step_name}: {e}")
                status.mark_completed(step_num, False)
                break
            
            # Add delay between steps
            time.sleep(1)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
