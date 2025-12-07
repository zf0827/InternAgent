import json
import openreview


client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
VENUE_ID = "ICLR.cc/2025/Conference"

def extract_reviews_fast(client, submission_id, reply_types=None):
    """
    Fetch directly by paper ID to avoid loading all submissions
    """
    if reply_types is None:
        reply_types = ["Official_Review", "Meta_Review", "Official_Comment"]
    
    try:
        # Directly fetch the specified paper's details
        submission = client.get_note(submission_id, details='replies')
        
        if not submission:
            print(f"× Cannot find submission: {submission_id}")
            return []
        
        replies_raw = submission.details.get("replies", [])
        matched_replies = []

        for reply in replies_raw:
            invitations = reply.get("invitations", [])
            for inv in invitations:
                for rtype in reply_types:
                    if inv.endswith(rtype):
                        matched_replies.append({
                            "reply_id": reply["id"],
                            "type": rtype,
                            "invitation": inv,
                            "signatures": reply.get("signatures", []),
                            "content": reply.get("content", {})
                        })
                        break  
        
        return matched_replies
        
    except Exception as e:
        print(f"× Failed to fetch paper {submission_id}: {e}")
        return []

def process_file_fast(input_file, output_file, venue_id):
    print(f"\n=== Fast processing {input_file} ===")

    with open(input_file, "r", encoding="utf-8") as f:
        papers = json.load(f)

    updated_papers = []

    for i, p in enumerate(papers, 1):
        pid = p["id"]
        print(f"  → [{i}/{len(papers)}] Fetching paper {pid}...")
        
        reviews = extract_reviews_fast(client, pid)
        p["reviews"] = reviews
        updated_papers.append(p)
        
        # Optional: pause every 10 papers to avoid overwhelming requests
        if i % 10 == 0:
            print(f"   Completed {i}/{len(papers)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_papers, f, ensure_ascii=False, indent=2)

    print(f"✔ Output saved to: {output_file}")

if __name__ == "__main__":
    file_pairs = [
        ("oral.json", "oral_with_reviews.json"),
        ("poster.json", "poster_with_reviews.json"), 
        ("spotlight.json", "spotlight_with_reviews.json"),
        ("reject.json", "reject_with_reviews.json"),
    ]

    for input_file, output_file in file_pairs:
        try:
            process_file_fast(input_file, output_file, VENUE_ID)
        except FileNotFoundError:
            print(f"⚠ Skipping (file not found): {input_file}")