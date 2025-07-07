# scripts/check_status.py
"""
Quick status checker for the daily pipeline.
Shows what's been completed and what's pending.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from daily_pipeline import PipelineStatus

def main():
    """Show current pipeline status"""
    print("🏇 UK HORSE RACING PIPELINE STATUS")
    print("=" * 40)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)
    
    status = PipelineStatus()
    
    print("\n📊 STEP STATUS:")
    for line in status.get_status_summary():
        print(line)
    
    # Count completed steps
    completed_count = sum(1 for step_info in status.steps.values() 
                         if step_info["completed"] and step_info["success"])
    total_count = len(status.steps)
    
    print(f"\n🎯 Progress: {completed_count}/{total_count} steps completed successfully")
    
    # Show next step
    next_step = status.get_next_step()
    if next_step:
        step_name = status.steps[next_step]["name"]
        print(f"⏭️  Next step: {next_step}. {step_name}")
    else:
        print("✅ All steps completed!")
    
    # Show any failed steps
    failed_steps = [step_num for step_num, step_info in status.steps.items() 
                   if step_info["completed"] and not step_info["success"]]
    
    if failed_steps:
        print(f"\n❌ Failed steps: {', '.join(failed_steps)}")
        print("💡 Use 'python run_individual_steps.py' to retry specific steps")

if __name__ == "__main__":
    main()
