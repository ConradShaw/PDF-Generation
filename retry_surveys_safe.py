import requests
import psycopg2
import json

# ----------------------------
# 1. Database connection info
# ----------------------------
DB_HOST = db.zsuzncnguhtvevivbxrn.supabase.co
DB_PORT = 5432
DB_NAME = postgres
DB_USER = postgres
DB_PASSWORD = "YOUR_DB_PASSWORD"

# ----------------------------
# 2. Your endpoint URL
# ----------------------------
PDF_ENDPOINT = "https://your-domain.com/generate-team-pdf"

# ----------------------------
# 3. Team info
# ----------------------------
COMPANY_NAME = "My Company"
TEAM_NAME = "Team Alpha"
DATE_STR = "2026-04-01"  # report date

# ----------------------------
# Connect to DB and fetch failed surveys
# ----------------------------
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)
cur = conn.cursor()

cur.execute("""
SELECT id, user_email, ordered_traits_json, ranks_json, distribution_data_json
FROM surveys
WHERE status = 'failed';
""")
rows = cur.fetchall()

individual_results = []
for row in rows:
    survey_id, user_email, ordered_traits_json, ranks_json, distribution_data_json = row
    try:
        individual_results.append({
            "id": survey_id,
            "user_email": user_email,
            "ordered_traits": json.loads(ordered_traits_json),
            "ranks": json.loads(ranks_json),
            "distribution_data": json.loads(distribution_data_json),
            "logo_path": "logo.png"  # optional, adjust if needed
        })
    except Exception as e:
        print(f"Skipping survey {survey_id} due to JSON decode error: {e}")

cur.close()
conn.close()

if not individual_results:
    print("No failed surveys found to retry.")
    exit(0)

# ----------------------------
# 4. Build payload and call endpoint
# ----------------------------
payload = {
    "company_name": COMPANY_NAME,
    "team_name": TEAM_NAME,
    "num_members": len(individual_results),
    "date_str": DATE_STR,
    "individual_results": individual_results
}

response = requests.post(PDF_ENDPOINT, json=payload)
if response.status_code != 200:
    print(f"Endpoint returned status {response.status_code}: {response.text}")
else:
    data = response.json()
    print("=== Retry Results ===")
    print(json.dumps(data, indent=2))
