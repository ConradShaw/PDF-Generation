#!/usr/bin/env python3

from supabase import create_client
import requests
import json

# --- CONFIG (paste your values here) ---
SUPABASE_URL = "https://zsuzncnguhtvevivbxrn.supabase.co"
SUPABASE_KEY = "PASTE_YOUR_SERVICE_ROLE_KEY_HERE"

PDF_ENDPOINT = "https://YOUR-SERVICE-URL/generate_team_pdf"
# Example:
# PDF_ENDPOINT = "https://your-cloud-run-url/generate_team_pdf"

# --- Init Supabase ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Fetching failed surveys...")

response = supabase.table("surveys") \
    .select("*") \
    .eq("status", "failed") \
    .execute()

rows = response.data

if not rows:
    print("No failed surveys found.")
    exit()

print(f"Found {len(rows)} failed surveys\n")

results_summary = []
skipped_surveys = []

# --- Process each survey ---
for survey in rows:
    survey_id = survey.get("id")
    user_email = survey.get("user_email", "unknown")

    # If your data is stored as JSON string, adjust this:
    survey_data = survey.get("survey_data_json")

    if isinstance(survey_data, str):
        survey_data = json.loads(survey_data)

    # --- Check required fields ---
    required_fields = ["ordered_traits", "ranks", "distribution_data"]
    if not all(field in survey_data and survey_data[field] for field in required_fields):
        skipped_surveys.append({
            "survey_id": survey_id,
            "reason": "Incomplete data"
        })
        print(f"Skipping {survey_id} — incomplete data")
        continue

    try:
        print(f"Retrying survey {survey_id}...")

        response = requests.post(PDF_ENDPOINT, json={
            "company_name": survey_data.get("company_name"),
            "team_name": survey_data.get("team_name"),
            "num_members": survey_data.get("num_members"),
            "date_str": survey_data.get("date_str"),
            "individual_results": [survey_data]
        })

        if response.status_code == 200 and response.json().get("success"):
            # Mark as processed
            supabase.table("surveys") \
                .update({"status": "processed"}) \
                .eq("id", survey_id) \
                .execute()

            results_summary.append({"survey_id": survey_id, "status": "success"})
            print(f"SUCCESS: {survey_id}")

        else:
            results_summary.append({"survey_id": survey_id, "status": "failed"})
            print(f"FAILED: {survey_id} → {response.text}")

    except Exception as e:
        results_summary.append({"survey_id": survey_id, "status": "failed"})
        print(f"ERROR: {survey_id} → {str(e)}")

# --- Summary ---
print("\n=== SUMMARY ===")
print(f"Successful: {len([r for r in results_summary if r['status']=='success'])}")
print(f"Failed: {len([r for r in results_summary if r['status']=='failed'])}")
print(f"Skipped: {len(skipped_surveys)}")

if skipped_surveys:
    print("\nSkipped surveys:")
    for s in skipped_surveys:
        print(s)


