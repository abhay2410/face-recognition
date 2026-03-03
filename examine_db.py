import asyncio
import database

async def show():
    conn = database._get_conn()
    cur = conn.cursor()
    try:
        # Check if door_name exists first (to be safe)
        cur.execute("SELECT TOP 5 id, employee_id, matched_at, distance, door_api_ok, door_name FROM access_log ORDER BY matched_at DESC")
        logs = cur.fetchall()
        print("\n--- Last 5 Access Logs ---")
        for log in logs:
            print(f"ID: {log[0]}, EmpID: {log[1]}, At: {log[2]}, Dist: {log[3]}, Ok: {log[4]}, Door: {log[5]}")
    except Exception as e:
        print(f"Error reading logs: {e}")
        # Fallback if door_name isn't there yet
        try:
            cur.execute("SELECT TOP 5 id, employee_id, matched_at, distance, door_api_ok FROM access_log ORDER BY matched_at DESC")
            logs = cur.fetchall()
            print("\n--- Last 5 Access Logs (Legacy Schema) ---")
            for log in logs:
                print(f"ID: {log[0]}, EmpID: {log[1]}, At: {log[2]}, Dist: {log[3]}, Ok: {log[4]}")
        except Exception as e2:
            print(f"Critical Error: {e2}")
    finally:
        conn.close()

if __name__ == "__main__":
    asyncio.run(show())
