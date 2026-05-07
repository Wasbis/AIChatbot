import re

text = """
## 3 Common Asset Recording Mistakes in Oil and Gas Plants and Their Solutions

18 December 2025|3 min read

Share

## 3 Common Asset Recording Mistakes in Oil and Gas Plants and Their Solutions

In the oil and gas industry...
### Closing Statement
Accurate asset records are the backbone...
Author : Jen Megah Bremanda Sembiring

#### Discover more insights
"""

author_match = re.search(r'Author\s*:\s*(.+)', text, re.IGNORECASE)
if author_match:
    print(f"Author found: {author_match.group(1).strip()}")

date_match = re.search(r'(\d{1,2}\s+[a-zA-Z]+\s+\d{4})', text)
if date_match:
    print(f"Date found: {date_match.group(1)}")
