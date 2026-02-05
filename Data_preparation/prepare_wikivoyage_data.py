#!/usr/bin/env python3
"""
Complete WikiVoyage Data Preparation Pipeline

This script:
1. Cleans WikiVoyage data (removes wiki markup, templates, links, etc.)
2. Filters out non-travel articles
3. Splits articles into structured sections
4. Outputs data in the format: {"id": "...", "title": "...", "section_name": "content", ...}
"""

import json
import re
from pathlib import Path
import sys

# Check for mwparserfromhell
try:
    import mwparserfromhell
    HAS_MWPARSER = True
except ImportError:
    HAS_MWPARSER = False
    print("Warning: mwparserfromhell not installed. Using regex-only mode.")


# ============================================================================
# COMMON WIKIVOYAGE SECTIONS
# ============================================================================

COMMON_SECTIONS = [
    'Understand',
    'Orientation',
    'History',
    'Climate',
    'Get in',
    'Fees and permits',
    'Get around',
    'See',
    'Do',
    'Buy',
    'Eat',
    'Drink',
    'Sleep',
    'Learn',
    'Work',
    'Connect',
    'Cope',
    'Stay safe',
    'Stay healthy',
    'Respect',
    'Go next',
]


# ============================================================================
# TEXT CLEANING FUNCTIONS
# ============================================================================

def clean_with_mwparser(text):
    """Clean text using mwparserfromhell library for robust wiki parsing."""
    if not text or not text.strip():
        return ""
    
    try:
        wikicode = mwparserfromhell.parse(text)
        
        # Remove all templates ({{...}})
        for template in wikicode.filter_templates():
            wikicode.remove(template)
        
        # Remove all wikilinks but keep the display text
        for wikilink in wikicode.filter_wikilinks():
            if wikilink.text:
                wikicode.replace(wikilink, wikilink.text)
            elif wikilink.title:
                wikicode.replace(wikilink, wikilink.title.strip_code())
        
        # Remove HTML tags and their content
        for tag in wikicode.filter_tags():
            wikicode.remove(tag)
        
        # Strip to plain text
        cleaned = wikicode.strip_code()
        
        # Additional cleanup with regex
        cleaned = clean_with_regex(cleaned, skip_templates_links=True)
        
        return cleaned
        
    except Exception:
        return clean_with_regex(text)


def clean_with_regex(text, skip_templates_links=False):
    """Clean wiki markup using regex patterns."""
    if not text or text.strip() == "":
        return ""
    
    if not skip_templates_links:
        # Remove tables
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        
        # Remove templates {{...}}
        max_iterations = 10
        for _ in range(max_iterations):
            prev_text = text
            text = re.sub(r'\{\{[^{}]*?\}\}', '', text)
            if prev_text == text:
                break
        
        text = re.sub(r'\{\{|\}\}', '', text)
        
        # Remove file/image references
        text = re.sub(r'\[\[File:.*?\]\]', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'\[\[Category:.*?\]\]', '', text, flags=re.IGNORECASE)
        
        # Convert wiki links [[link|text]] to just text
        text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
        text = re.sub(r'\[\[|\]\]', '', text)
    
    # Remove coordinates and geo data
    text = re.sub(r'\{coord\|.*?\}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<coordinates>.*?</coordinates>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'\d+°\s*\d*\'?\s*[NSEW]\s+\d+°\s*\d*\'?\s*[NSEW]', '', text)
    text = re.sub(r'\d+\.\d+°?\s*[NSEW],?\s*\d+\.\d+°?\s*[NSEW]', '', text)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove reference tags
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<ref[^>]*/?\>', '', text, flags=re.IGNORECASE)
    
    # Remove gallery tags
    text = re.sub(r'<gallery[^>]*>.*?</gallery>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove common wiki artifacts
    text = re.sub(r"'''|''", '', text)  # Bold/italic markup
    text = re.sub(r'^[#\*:;]+\s*', '', text, flags=re.MULTILINE)  # List markers
    text = re.sub(r'^=+\s*(.*?)\s*=+$', r'\1', text, flags=re.MULTILINE)  # Headers
    
    # Remove external URLs
    text = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    text = text.strip()
    
    return text


def clean_text(text):
    """Main cleaning function - uses mwparserfromhell if available, else regex."""
    if HAS_MWPARSER:
        return clean_with_mwparser(text)
    else:
        return clean_with_regex(text)


# ============================================================================
# ARTICLE FILTERING
# ============================================================================

def is_valid_travel_article(article):
    """
    Filter out non-travel articles.
    
    Returns True if the article appears to be a valid travel destination.
    """
    title = article.get('title', '')
    text = article.get('text', '')
    
    # Filter 1: Must have non-empty text
    if not text or len(text.strip()) < 100:
        return False
    
    # Filter 2: Year pages (4 digits)
    if re.match(r'^\d{4}$', title):
        return False
    
    # Filter 3: Meta/nonsense titles
    junk_patterns = [
        r'^\d+liner$',  # "1liner", "2liner"
        r'^[\d\s\+]+$',  # "7 2", "7+2"
        r'^\d+\s*[\+\-\*\/]\s*\d+$',  # Math expressions
    ]
    
    for pattern in junk_patterns:
        if re.match(pattern, title, re.IGNORECASE):
            return False
    
    # Filter 4: Event/sports pages
    event_keywords = [
        'world cup', 'olympics', 'fifa', 'championship',
        'tournament', 'conference', 'summit'
    ]
    
    title_lower = title.lower()
    if any(keyword in title_lower for keyword in event_keywords):
        if re.match(r'^\d{4}', title):  # Starts with year
            return False
    
    # Filter 5: Disambiguation pages
    if '(disambiguation)' in title_lower:
        return False
    
    return True


# ============================================================================
# SECTION EXTRACTION
# ============================================================================

def extract_sections_robust(text):
    """
    Extract sections from cleaned WikiVoyage text.
    
    Returns dict with section names as keys and content as values.
    """
    if not text or len(text.strip()) < 50:
        return {}
    
    sections = {}
    
    # Create pattern that matches any of the common sections
    section_names = '|'.join(re.escape(s) for s in COMMON_SECTIONS)
    pattern = rf'(?=^(?:{section_names})\.?\s*$)'
    
    # Split text into chunks
    chunks = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Process chunks
    intro_parts = []
    
    for chunk in chunks:
        if not chunk.strip():
            continue
        
        # Check if chunk starts with a section header
        lines = chunk.strip().split('\n', 1)
        first_line = lines[0].strip().rstrip('.')
        
        # Check if this is a known section header
        section_found = None
        for section_name in COMMON_SECTIONS:
            if first_line.lower() == section_name.lower():
                section_found = section_name
                break
        
        if section_found:
            # Get content after the header
            content = lines[1] if len(lines) > 1 else ''
            content = content.strip()
            content = clean_section_content(content)
            
            if content:
                # Convert to snake_case field name
                field_name = section_found.lower().replace(' ', '_')
                sections[field_name] = content
        else:
            # This is intro content
            intro_parts.append(chunk.strip())
    
    # Combine intro parts
    if intro_parts:
        intro = '\n\n'.join(intro_parts).strip()
        intro = clean_section_content(intro)
        if intro:
            sections['intro'] = intro
    
    return sections


def clean_section_content(content):
    """Remove stray section headers from content."""
    if not content:
        return ''
    
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip().rstrip('.')
        
        # Check if this line is just a section header
        is_header = False
        for section_name in COMMON_SECTIONS:
            if stripped.lower() == section_name.lower():
                is_header = True
                break
        
        if not is_header:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines).strip()
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    return result


# ============================================================================
# FILE PROCESSING
# ============================================================================

def process_json_file(input_path, output_path):
    """Process a single JSON file: clean, filter, and split sections."""
    
    processed_articles = []
    total_count = 0
    filtered_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                article = json.loads(line)
                total_count += 1
                
                # Clean the text
                original_text = article.get('text', '')
                cleaned_text = clean_text(original_text)
                
                # Update article with cleaned text
                article['text'] = cleaned_text
                
                # Filter out junk articles
                if not is_valid_travel_article(article):
                    filtered_count += 1
                    continue
                
                # Extract sections
                sections = extract_sections_robust(cleaned_text)
                
                # Create final article structure
                final_article = {
                    'id': article['id'],
                    'title': article['title'],
                }
                
                # Add all sections as top-level fields
                final_article.update(sections)
                
                processed_articles.append(final_article)
                
            except json.JSONDecodeError as e:
                print(f"  Error parsing JSON at line {line_num} in {input_path.name}: {e}")
                filtered_count += 1
                continue
            except Exception as e:
                print(f"  Error processing line {line_num} in {input_path.name}: {e}")
                filtered_count += 1
                continue
    
    # Write processed articles
    if processed_articles:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for article in processed_articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
    
    return len(processed_articles), filtered_count, total_count


def process_directory(input_dir, output_dir):
    """Process all wiki files in the directory structure."""
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    total_kept = 0
    total_filtered = 0
    total_processed = 0
    file_count = 0
    
    # Find all wiki files
    wiki_files = sorted(list(input_path.rglob('wiki_*')))
    
    if not wiki_files:
        print(f"❌ No wiki files found in {input_dir}")
        return
    
    print(f"Found {len(wiki_files)} files to process")
    print(f"Using {'mwparserfromhell' if HAS_MWPARSER else 'regex-only'} parser")
    print()
    
    for wiki_file in wiki_files:
        # Create corresponding output path
        relative_path = wiki_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        print(f"Processing {relative_path}...", end=' ')
        
        try:
            kept, filtered, processed = process_json_file(wiki_file, output_file)
            total_kept += kept
            total_filtered += filtered
            total_processed += processed
            file_count += 1
            
            print(f"✓ (kept {kept}/{processed}, filtered {filtered})")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    print()
    print("=" * 70)
    print(f"✅ Complete!")
    print(f"  Processed {file_count} files")
    print(f"  Total articles processed: {total_processed}")
    print(f"  Articles kept: {total_kept} ({total_kept/total_processed*100:.1f}%)")
    print(f"  Articles filtered: {total_filtered} ({total_filtered/total_processed*100:.1f}%)")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Default paths
    input_dir = "enwikivoyage-latest-pages-articles"
    output_dir = "enwikivoyage-sectioned"
    
    # Allow command-line arguments
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    print("WikiVoyage Complete Data Preparation Pipeline")
    print("=" * 70)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)
    print()
    
    if not Path(input_dir).exists():
        print(f"❌ Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)
    
    process_directory(input_dir, output_dir)
