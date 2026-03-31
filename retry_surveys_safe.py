import requests
import psycopg2
import json
import logging
from datetime import datetime
import time
import os

# ----------------------------
# 1. Logging setup
# ----------------------------
LOG_FILE = "retry_surveys.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ----------------------------
# 2. Database connection info
# ----------------------------
DB_HOST = "db.zsuzncnguhtvevivbxrn.supabase.co"
DB_PORT = 5432
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "daiG12tuoNwY9IO4"

# ----------------------------
# 3. Endpoint and team info
# ----------------------------
PDF_ENDPOINT = "https://www.shawsight.com/hr/performance-ratings"
COMPANY_NAME = "ShawSight Pty Ltd"
TEAM_NAME = "Founder"
DATE_STR = datetime.now().strftime("%Y-%m-%d")  # report date

# ----------------------------
# 4. Load previous failed surveys (if any)
# ----------------------------
FAILED_FILE = "retry_failed_surveys.json"
previous_failed_ids = []

if os.path.exists(FAILED_FILE):
    with open(FAILED_FILE, "r") as f:
        try:
            previous_failed_ids = json.load(f)
            logging.info(f"Loaded {len(previous_failed_ids)} previously failed surveys.")
        except Exception as e:
            logging.warning(f"Failed to load previous failed surveys: {e}")
            previous_failed_ids = []

# ----------------------------
# 5. Connect to DB and fetch surveys
# ----------------------------
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
    logging.info("Connected to database successfully.")
except Exception as e:
    logging.error(f"Database connection failed: {e}")
    exit(1)

# Determine which surveys to fetch
if previous_failed_ids:
    placeholders = ",".join(["%s"] * len(previous_failed_ids))
    query = f"""
        SELECT id, user_email, ordered_traits_json, ranks_json, distribution_data_json
        FROM surveys
        WHERE id IN ({placeholders});
    """
    cur.execute(query, previous_failed_ids)
else:
    # fetch all failed surveys
    cur.execute("""
        SELECT id, user_email, ordered_traits_json, ranks_json, distribution_data_json
        FROM surveys
        WHERE status = 'failed';
    """)

rows = cur.fetchall()
logging.info(f"Fetched {len(rows)} surveys for retry.")

individual_results = []

def safe_load(json_str, default, survey_id, field_name):
    try:
        return json.loads(json_str) if json_str else default
    except Exception as e:
        logging.warning(f"Survey {survey_id}: failed to parse {field_name} - {e}")
        return default

for row in rows:
    survey_id, user_email, ordered_traits_json, ranks_json, distribution_data_json = row
    individual_results.append({
        "id": survey_id,
        "user_email": user_email,
        "ordered_traits": safe_load(ordered_traits_json, [], survey_id, "ordered_traits_json"),
        "ranks": safe_load(ranks_json, [], survey_id, "ranks_json"),
        "distribution_data": safe_load(distribution_data_json, {}, survey_id, "distribution_data_json"),
        "logo_path": "logo.png"
    })

cur.close()
conn.close()
logging.info("Database connection closed.")

if not individual_results:
    logging.info("No surveys found to retry.")
    print("No surveys found to retry.")
    exit(0)

# ----------------------------
# 6. Build payload
# ----------------------------
payload = {
    "company_name": COMPANY_NAME,
    "team_name": TEAM_NAME,
    "num_members": len(individual_results),
    "date_str": DATE_STR,
    "individual_results": individual_results
}

# ----------------------------
# 7. Endpoint call with retries
# ----------------------------
MAX_RETRIES = 3
RETRY_DELAY = 5
survey_results = []

for attempt in range(1, MAX_RETRIES + 1):
    try:
        response = requests.post(PDF_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        for result in data.get("results", []):
            survey_id = result.get("id")
            status = result.get("status", "failed")
            survey_results.append({"id": survey_id, "status": status})

        logging.info(f"Endpoint returned results on attempt {attempt}.")
        logging.info(json.dumps(data, indent=2))
        print("=== Retry Results ===")
        print(json.dumps(data, indent=2))
        break
    except requests.RequestException as e:
        logging.warning(f"Attempt {attempt}: Error contacting endpoint: {e}")
        print(f"Attempt {attempt}: Error contacting endpoint: {e}")
    except json.JSONDecodeError:
        logging.warning(f"Attempt {attempt}: Endpoint did not return valid JSON")
        print(f"Attempt {attempt}: Endpoint did not return valid JSON")

    if attempt < MAX_RETRIES:
        logging.info(f"Retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)
    else:
        logging.error("All retry attempts failed.")
        print("All retry attempts failed.")
        for s in individual_results:
            survey_results.append({"id": s["id"], "status": "failed"})

# ----------------------------
# 8. Summary & update failed file
# ----------------------------
total = len(survey_results)
success_count = sum(1 for s in survey_results if s["status"] == "success")
failed_surveys = [s["id"] for s in survey_results if s["status"] != "success"]
failed_count = len(failed_surveys)

print("\n=== Summary ===")
print(f"Total surveys processed: {total}")
print(f"Successfully retried: {success_count}")
print(f"Failed: {failed_count}")

logging.info("=== Summary ===")
logging.info(f"Total surveys processed: {total}")
logging.info(f"Successfully retried: {success_count}")
logging.info(f"Failed: {failed_count}")

print("\nPer-survey status:")
for s in survey_results:
    print(f"Survey ID {s['id']}: {s['status']}")

# Save failed survey IDs for next retry
with open(FAILED_FILE, "w") as f:
    json.dump(failed_surveys, f)
logging.info(f"{failed_count} surveys saved for next retry.")


