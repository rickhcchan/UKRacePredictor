#!/usr/bin/env python3
"""
Test script for AI integration functionality
Tests the cargowise domain prompts integration with UK Race Predictor
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from prompts.ai_integration import ai_integration
from prompts.prompt_manager import prompt_manager

def test_prompt_manager():
    """Test prompt manager functionality"""
    print("🧪 Testing Prompt Manager...")
    
    # List available prompts
    available = prompt_manager.list_available_prompts()
    print(f"Available domains: {list(available.keys())}")
    
    if 'cargowise' in available:
        print(f"Cargowise prompts: {available['cargowise']}")
    else:
        print("❌ Cargowise domain not found")
        return False
    
    # Test loading a prompt
    analysis_prompt = prompt_manager.load_prompt("cargowise", "analysis")
    if analysis_prompt:
        print("✓ Analysis prompt loaded successfully")
        print(f"Prompt length: {len(analysis_prompt)} characters")
    else:
        print("❌ Failed to load analysis prompt")
        return False
    
    return True

def test_ai_integration():
    """Test AI integration functionality"""
    print("\n🤖 Testing AI Integration...")
    
    # Get status
    status = ai_integration.get_prompt_status()
    print(f"AI Integration Status: {status}")
    
    # Test race analysis
    sample_race_data = {
        'course': 'Ascot',
        'time': '15:30',
        'distance': '1m 2f',
        'going': 'Good',
        'race_type': 'Handicap',
        'field_size': 12,
        'horses': [
            {
                'name': 'Thunder Bay',
                'age': 4,
                'weight': '9-7',
                'jockey': 'Ryan Moore',
                'trainer': 'Aidan O\'Brien',
                'recent_form': '121'
            },
            {
                'name': 'Lightning Strike',
                'age': 5,
                'weight': '9-3',
                'jockey': 'William Buick',
                'trainer': 'Charlie Appleby',
                'recent_form': '341'
            }
        ]
    }
    
    analysis = ai_integration.analyze_race_data(sample_race_data)
    if analysis:
        print("✓ Race analysis generated successfully")
        print(f"Analysis preview (first 200 chars): {analysis[:200]}...")
    else:
        print("❌ Failed to generate race analysis")
        return False
    
    # Test prediction enhancement
    sample_predictions = {
        'Thunder Bay': {'probability': 0.25, 'confidence': 'high'},
        'Lightning Strike': {'probability': 0.18, 'confidence': 'medium'}
    }
    
    sample_context = {
        'course': 'Ascot',
        'time': '15:30',
        'distance': '1m 2f',
        'field_size': 12,
        'going': 'Good'
    }
    
    enhanced = ai_integration.enhance_predictions(sample_predictions, sample_context)
    if enhanced and 'ai_enhancement' in enhanced:
        print("✓ Prediction enhancement completed successfully")
    else:
        print("❌ Failed to enhance predictions")
        return False
    
    # Test strategy recommendations
    sample_horses = [
        {
            'name': 'Thunder Bay',
            'probability': 0.25,
            'selection_reason': 'High probability and good recent form'
        }
    ]
    
    sample_strategy_context = {
        'course': 'Ascot',
        'time': '15:30',
        'risk_profile': 'Moderate',
        'capital_allocation': 'Standard',
        'market_state': 'Normal'
    }
    
    recommendations = ai_integration.generate_strategy_recommendations(
        sample_horses, sample_strategy_context
    )
    if recommendations:
        print("✓ Strategy recommendations generated successfully")
        print(f"Recommendations preview (first 200 chars): {recommendations[:200]}...")
    else:
        print("❌ Failed to generate strategy recommendations")
        return False
    
    return True

def test_prompt_formatting():
    """Test prompt template formatting"""
    print("\n📝 Testing Prompt Formatting...")
    
    # Test cargowise analysis prompt
    race_data = {
        'course': 'Newmarket',
        'distance': '1m',
        'going': 'Firm',
        'field_size': 8,
        'race_type': 'Maiden',
        'horses': [
            {
                'name': 'Test Horse',
                'age': 3,
                'weight': '9-0',
                'jockey': 'Test Jockey',
                'trainer': 'Test Trainer'
            }
        ]
    }
    
    formatted_prompt = prompt_manager.get_cargowise_analysis_prompt(race_data)
    if formatted_prompt:
        print("✓ Analysis prompt formatted successfully")
        print(f"Contains course name: {'Newmarket' in formatted_prompt}")
        print(f"Contains distance: {'1m' in formatted_prompt}")
    else:
        print("❌ Failed to format analysis prompt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🏇 UK Race Predictor - AI Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Prompt Manager", test_prompt_manager),
        ("AI Integration", test_ai_integration),
        ("Prompt Formatting", test_prompt_formatting)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AI integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)