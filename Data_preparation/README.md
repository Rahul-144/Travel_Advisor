# WikiVoyage Data Preparation

This directory contains the complete pipeline for preparing WikiVoyage data for the Travel Advisor project.

## Main Script

**`prepare_wikivoyage_data.py`** - Complete data preparation pipeline that:
1. Cleans WikiVoyage data (removes wiki markup, templates, links, coordinates, etc.)
2. Filters out non-travel articles (year pages, events, disambiguation pages, etc.)
3. Splits articles into structured sections (intro, get_in, see, do, eat, drink, etc.)
4. Outputs data in the required format with sections as top-level fields

## Output Format

Each processed article is a JSON object with this structure:
```json
{
  "id": "123",
  "title": "City Name",
  "intro": "Introductory text...",
  "understand": "Background information...",
  "get_in": "How to get there...",
  "get_around": "Transportation info...",
  "see": "Things to see...",
  "do": "Activities...",
  "buy": "Shopping info...",
  "eat": "Dining options...",
  "drink": "Bars and nightlife...",
  "sleep": "Accommodation...",
  "stay_safe": "Safety information...",
  "go_next": "Nearby destinations..."
}
```

Not all sections will be present in every article. Sections are only included if they have content.

## Usage

### Basic Usage
```bash
python3 prepare_wikivoyage_data.py <input_dir> <output_dir>
```

### Example
```bash
# Process extracted WikiVoyage data
python3 prepare_wikivoyage_data.py enwikivoyage-latest-pages-articles enwikivoyage-sectioned
```

### Default Behavior
If no arguments are provided:
- Input directory: `enwikivoyage-latest-pages-articles`
- Output directory: `enwikivoyage-sectioned`

## Dependencies

### Required
- Python 3.6+
- Standard library modules (json, re, pathlib, sys)

### Optional
- `mwparserfromhell` - For more robust MediaWiki parsing (highly recommended)
  ```bash
  pip install mwparserfromhell
  ```
  
If `mwparserfromhell` is not installed, the script falls back to regex-only mode.

## Input Data

The script expects WikiVoyage data extracted by WikiExtractor in JSONL format.
Each line should be a JSON object with at least:
```json
{"id": "...", "title": "...", "text": "..."}
```

## Processing Steps

1. **Cleaning**: Removes MediaWiki markup including:
   - Wiki links `[[...]]`
   - Templates `{{...}}`
   - HTML tags and comments
   - Tables
   - Coordinates and geo data
   - References and citations
   - Bold/italic markup
   - List markers

2. **Filtering**: Removes non-travel content:
   - Year pages (e.g., "2020")
   - Event pages (e.g., "2018 FIFA World Cup")
   - Disambiguation pages
   - Articles with very short content (<100 characters)
   - Meta/junk pages

3. **Section Splitting**: Extracts standard WikiVoyage sections:
   - Intro (content before first section)
   - Understand
   - Get in
   - Get around
   - See, Do, Buy, Eat, Drink
   - Sleep, Stay safe, Stay healthy
   - Go next
   - And more...

## Output

- Maintains the same directory structure as input
- Creates one output file per input file
- Each output file contains one JSON object per line (JSONL format)
- Only includes valid travel articles with cleaned, structured content

## Statistics

The script provides detailed statistics during processing:
- Number of files processed
- Total articles processed
- Articles kept vs. filtered
- Percentage breakdown

## Note

This is the unified version that replaces the previous scripts:
- ~~`clean_wikivoyage.py`~~ (removed)
- ~~`clean_wikivoyage_enhanced.py`~~ (removed)
- ~~`clean_wikivoyage_final.py`~~ (removed)
- ~~`split_sections.py`~~ (removed)
