import asyncio
import os
import sys

# Set up the path so imports from your submodules work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the async functions from both scripts
from run_test.generate_singlehop_test_data import generate_test_data
from generate_multihop_test_data import generate_multihop_test_data 

async def main():
    print("\n" + "=" * 30)
    print(" STARTING SINGLE-HOP GENERATION ".center(30, "="))
    print("\n" + "=" * 30)
    await generate_test_data()
    print("Single-hop generation complete.\n")
    
    print("\n" + "=" * 30)
    print(" STARTING MULTI-HOP GENERATION ".center(30, "="))
    print("\n" + "=" * 30)
    await generate_multihop_test_data()
    print("Multi-hop generation complete.\n")

if __name__ == "__main__":
    asyncio.run(main())
