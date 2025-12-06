import openreview
import json
from datetime import datetime

client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
)

def convert_timestamp_to_date(timestamp):
    return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d') if timestamp else None

def extract_submission_info(sub):
    return {
        'id': sub.id,
        'title': sub.content['title']['value'],
        'abstract': sub.content['abstract']['value'],
        'keywords': sub.content['keywords']['value'],
        'primary_area': sub.content['primary_area']['value'],
        'TLDR': sub.content['TLDR']['value'] if 'TLDR' in sub.content else "",
        'creation_date': convert_timestamp_to_date(sub.cdate),
        'original_date': convert_timestamp_to_date(sub.odate),
        'modification_date': convert_timestamp_to_date(sub.mdate),
        'venue': sub.content.get('venue', {}).get('value', ""),
        'forum_link': f"https://openreview.net/forum?id={sub.id}",
        'pdf_link': f"https://openreview.net/pdf?id={sub.id}"
    }

def get_submissions_by_decision(client, venue_id, decision_keyword):
    """
    decision_keyword: one of ["reject", "poster", "spotlight", "oral"]
    """
    # Get all submissions
    all_submissions = client.get_all_notes(invitation=f'{venue_id}/-/Submission')
    matched = []

    print("=== Get Paper Decision ===")
    decision_map = {}
    
    sample_submissions = all_submissions[:2000]  
    
    for sub in sample_submissions:
        try:
            print(f"\n Check Submission: {sub.id}")
            forum_notes = client.get_notes(forum=sub.id)
            
            for note in forum_notes:
                if hasattr(note, 'content'):
                    content_str = str(note.content).lower()
                    if 'decision' in content_str and 'paper' in content_str:
                        print(f"  Find Paper Decision Notes: {note.id}")
                        print(f"  Content_keys: {list(note.content.keys())}")
                        
                        decision_value = ""
                        for key, value in note.content.items():
                            if 'decision' in key.lower():
                                if isinstance(value, dict) and 'value' in value:
                                    decision_value = value['value']
                                else:
                                    decision_value = str(value)
                                break
                        
                        if decision_value:
                            decision_map[sub.id] = decision_value.lower()
                            print(f"  Decision: {decision_value}")
                            break  
            
        except Exception as e:
            print(f"  error: {e}")
            continue
    
    
    # Decision statistics
    decision_counts = {}
    for decision_value in decision_map.values():
        decision_counts[decision_value] = decision_counts.get(decision_value, 0) + 1

    for decision_value, count in decision_counts.items():
        print(f"'{decision_value}': {count} papers")

    # Matching logic
    for sub in all_submissions:
        decision_value = decision_map.get(sub.id, '')
        
        if decision_keyword.lower() == "reject":
            if "reject" in decision_value:
                matched.append(sub)

    return matched

if __name__ == '__main__':
    venue_id = 'ICLR.cc/2025/Conference'
    print("Fetching all decisions...")

    categories = {
        "reject": "reject.json"
    }

    for decision, filename in categories.items():
        subs = get_submissions_by_decision(client, venue_id, decision)
        infos = [extract_submission_info(s) for s in subs]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(infos, f, ensure_ascii=False, indent=2)

        print(f"{decision.capitalize():10s}: {len(infos):4d} saved → {filename}")

    print("\nDone! All decision files generated.")