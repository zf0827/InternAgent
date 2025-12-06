import openreview
import json
from datetime import datetime


client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net'
)


def convert_timestamp_to_date(timestamp):
    return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d') if timestamp else None

# Extract the required information
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

# -------------------------------
# get (Poster / Spotlight / Oral)
# -------------------------------
def get_submissions_by_decision(client, venue_id, decision_keyword):
    """
    decision_keyword: one of ["poster", "spotlight", "oral"]
    """
    # 获取所有提交
    all_submissions = client.get_all_notes(invitation=f'{venue_id}/-/Submission')
    matched = []

    
    for sub in all_submissions:
        venue_value = sub.content.get("venue", {}).get("value", "").lower()
        if decision_keyword.lower() in venue_value:
            matched.append(sub)

    return matched

if __name__ == '__main__':

    venue_id = 'ICLR.cc/2025/Conference'

    print("Fetching all decisions...")

    categories = {
        "poster": "poster.json",
        "spotlight": "spotlight.json",
        "oral": "oral.json"
    }

    for decision, filename in categories.items():
        subs = get_submissions_by_decision(client, venue_id, decision)
        infos = [extract_submission_info(s) for s in subs]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(infos, f, ensure_ascii=False, indent=2)

        print(f"{decision.capitalize():10s}: {len(infos):4d} saved → {filename}")

    print("\nDone! All decision files generated.")