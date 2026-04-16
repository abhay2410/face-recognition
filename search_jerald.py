import asyncio
import database

async def main():
    try:
        # Assuming database has a function to connect or uses context managers
        # This is based on typical patterns in similar projects
        emps = await database.get_all_multi_embeddings()
        for e in emps:
            if "Jerald" in e["name"] or "Nepoleon" in e["name"]:
                print(f"MATCH: '{e['name']}' (ID: {e['id']})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
