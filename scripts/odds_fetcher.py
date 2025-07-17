"""
Module for fetching live odds from attheraces.com
"""
import json
import re
import os
from datetime import datetime
import pandas as pd

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    sync_playwright = None
    PlaywrightTimeoutError = Exception

def load_course_mappings():
    """Load course name mappings from JSON file"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'course_mappings.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Return empty dict if file not found or invalid JSON
        return {}

def normalize_course_name(course):
    """Normalize course name for URL using mappings"""
    course_lower = course.lower().strip()
    mappings = load_course_mappings()
    
    # Check direct mappings first (case-insensitive)
    if course_lower in mappings:
        return mappings[course_lower]
    
    # Check partial matches
    for key, value in mappings.items():
        if key in course_lower:
            return value
    
    # For unmapped courses, try to preserve proper case
    # Split on spaces and capitalize each word
    words = course.strip().split()
    if len(words) > 1:
        # Multi-word course names: capitalize each word
        return "-".join(word.capitalize() for word in words)
    else:
        # Single word: just capitalize
        return course.strip().capitalize()

def convert_to_24hr(time_str):
    """Convert time to 24hr format for URL"""
    try:
        # First try HH:MM format
        dt = datetime.strptime(time_str.strip(), "%H:%M")
        return dt.strftime("%H%M")
    except:
        try:
            # Then try 12-hour format
            dt = datetime.strptime(time_str.strip(), "%I:%M")
            if dt.hour < 12:
                dt = dt.replace(hour=dt.hour + 12)
            return dt.strftime("%H%M")
        except Exception:
            print(f"⚠️ Invalid time format: {time_str}")
            return "0000"

def reconstruct_horse_name(raw_name):
    """
    Try to reconstruct proper horse name from ID-style name
    e.g., 'roguesupremacy' -> 'Rogue Supremacy'
    """
    if not raw_name:
        return "Unknown"
    
    # Common patterns for splitting horse names
    # Look for obvious word boundaries
    import re
    
    # Add space before capital letters (except first)
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_name)
    
    # Add space before numbers
    name = re.sub(r'([a-z])([0-9])', r'\1 \2', name)
    
    # Title case each word
    words = name.split()
    
    # Try to intelligently split common compound words
    expanded_words = []
    for word in words:
        # Look for common word patterns that should be split
        word_lower = word.lower()
        
        # Some common horse name patterns
        if len(word) > 8:  # Long words might be compounds
            # Try to find obvious breaks
            for pattern in ['supremacy', 'master', 'king', 'queen', 'star', 'gold', 'silver', 'black', 'white', 'red']:
                if word_lower.endswith(pattern) and len(word_lower) > len(pattern):
                    prefix = word_lower[:-len(pattern)]
                    if len(prefix) > 2:  # Only split if prefix is meaningful
                        expanded_words.extend([prefix.title(), pattern.title()])
                        break
            else:
                expanded_words.append(word.title())
        else:
            expanded_words.append(word.title())
    
    return ' '.join(expanded_words)

def normalize_for_matching(name):
    """Normalize horse name for matching (remove spaces, lowercase)"""
    return re.sub(r'[^a-z0-9]', '', name.lower())

def find_best_horse_match(target_name, odds_dict):
    """Find the best matching horse name in odds_dict for target_name"""
    if target_name in odds_dict:
        return target_name
    
    # Normalize target name for matching
    target_normalized = normalize_for_matching(target_name)
    
    # Try to find exact normalized match
    for odds_horse in odds_dict.keys():
        if normalize_for_matching(odds_horse) == target_normalized:
            return odds_horse
    
    # Try partial matching
    for odds_horse in odds_dict.keys():
        odds_normalized = normalize_for_matching(odds_horse)
        if (target_normalized in odds_normalized and len(target_normalized) > 4) or \
           (odds_normalized in target_normalized and len(odds_normalized) > 4):
            return odds_horse
    
    return None

def get_race_odds(course, date_str, time_str, context=None):
    """Fetch odds for a specific race"""
    if not PLAYWRIGHT_AVAILABLE:
        print("⚠️ Playwright not available - cannot fetch odds")
        return {}
    
    should_close_context = False
    
    if context is None:
        p = sync_playwright().start()
        browser = p.chromium.launch(headless=False)  # Use non-headless for compatibility
        context = browser.new_context(viewport={"width": 1280, "height": 2000})
        should_close_context = True
    
    try:
        course_norm = normalize_course_name(course)
        time_norm = convert_to_24hr(time_str)
        
        # Convert date to correct format if needed
        if isinstance(date_str, str) and len(date_str.split('-')) == 3:
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                # attheraces.com uses format: "17-July-2025" (proper case month)
                date_formatted = date_obj.strftime("%d-%B-%Y")  # "17-July-2025"
            except:
                date_formatted = date_str
        else:
            date_formatted = date_str
            
        url = f"https://www.attheraces.com/racecard/{course_norm}/{date_formatted}/{time_norm}"
        
        page = context.new_page()
        odds_data = {}
        
        try:
            page.goto(url, timeout=60000)
            # Use the same scroll and wait approach as original scraper
            page.mouse.wheel(0, 5000)
            page.mouse.wheel(5000, 0)
            
            # Wait for the selector with shorter timeout like original
            page.wait_for_selector("div.odds-grid__row--horse", timeout=3000)
            rows = page.query_selector_all("div.odds-grid__row--horse")
            
            for row in rows:
                # Extract horse name from ID attribute
                horse_div = row.query_selector("div")
                horse_id = horse_div.get_attribute("id") if horse_div else ""
                raw_horse_name = horse_id.split("-")[0].strip() if horse_id else ""
                
                # Try to get a better horse name from the visible text
                horse_text_el = row.query_selector("div[class*='name'] span, div span[class*='name'], .horse-name, [data-horse-name]")
                if horse_text_el:
                    horse = horse_text_el.inner_text().strip()
                elif raw_horse_name:
                    # If no visible text, try to reconstruct proper spacing from ID
                    horse = reconstruct_horse_name(raw_horse_name)
                else:
                    horse = "Unknown"
                
                odds_el = row.query_selector("span.odds-value--decimal")
                odds = odds_el.inner_text().strip() if odds_el else "N/A"
                
                if horse != "Unknown" and horse and odds != "N/A":
                    odds_data[horse] = odds
                    
        except PlaywrightTimeoutError:
            print(f"⏰ Timeout fetching odds for {course} {time_str}")
        except Exception as e:
            print(f"❌ Error fetching odds: {e}")
        finally:
            page.close()
            
        return odds_data
        
    finally:
        if should_close_context:
            context.browser.close()
            p.stop()

def format_horse_with_odds(horse_name, odds_dict):
    """Format horse name with odds if available"""
    # Use the improved matching function
    matched_horse = find_best_horse_match(horse_name, odds_dict)
    
    if matched_horse:
        odds = odds_dict[matched_horse]
        return f"{horse_name} ({odds})"
    
    return f"{horse_name} (N/A)"

def scrape_odds_for_race(context, course, date_str, time_str):
    """
    Legacy function for compatibility with existing scraper
    Returns DataFrame format like the original script
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("⚠️ Playwright not available - cannot fetch odds")
        return pd.DataFrame()
    
    odds_dict = get_race_odds(course, date_str, time_str, context)
    
    odds_data = []
    for horse, odds in odds_dict.items():
        odds_data.append({
            "course": course,
            "time": convert_to_24hr(time_str),
            "horse": horse,
            "odds": odds
        })
    
    df = pd.DataFrame(odds_data)
    if not df.empty:
        df['odds_val'] = pd.to_numeric(df['odds'], errors='coerce').fillna(999)
        df = df.sort_values(by='odds_val').drop_duplicates(subset=['course', 'time', 'horse'], keep='first')
        df.drop(columns='odds_val', inplace=True)
        
        print(f"\n🎯 Odds scraped for {course} {time_str}:")
        print(df[['course', 'time', 'horse', 'odds']])
    
    return df

def get_possible_date_formats(date_str):
    """Get different possible date formats for attheraces.com URLs"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        formats = [
            date_obj.strftime("%d-%B-%Y").lower(),  # "17-july-2025"
            date_obj.strftime("%d-%B-%Y"),          # "17-July-2025"
            date_obj.strftime("%Y-%m-%d"),          # "2025-07-17"
            date_obj.strftime("%d-%m-%Y"),          # "17-07-2025"
            date_obj.strftime("%d%m%Y"),            # "17072025"
            date_obj.strftime("%Y%m%d"),            # "20250717"
        ]
        return formats
    except:
        return [date_str]

def test_race_url_formats(course, date_str, time_str, context):
    """Test different URL formats to find the correct one"""
    course_norm = normalize_course_name(course)
    time_norm = convert_to_24hr(time_str)
    
    possible_dates = get_possible_date_formats(date_str)
    
    for date_format in possible_dates:
        url = f"https://www.attheraces.com/racecard/{course_norm}/{date_format}/{time_norm}"
        print(f"🔗 Testing URL: {url}")
        
        page = context.new_page()
        try:
            response = page.goto(url, timeout=10000)
            if response and response.status == 200:
                # Check if the page has the expected content
                if page.query_selector("div.odds-grid__row--horse"):
                    print(f"✅ Found valid URL format: {date_format}")
                    page.close()
                    return date_format
                elif "race" in page.url.lower():
                    print(f"⚠️ Page found but no odds grid: {date_format}")
            else:
                print(f"❌ HTTP {response.status if response else 'No response'}: {date_format}")
        except Exception as e:
            print(f"❌ Error with {date_format}: {str(e)[:100]}")
        finally:
            page.close()
    
    print("❌ No valid URL format found")
    return None
