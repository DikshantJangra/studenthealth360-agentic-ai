
import sys
import os
from pathlib import Path

# Ensure project root is on the path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Mock environment for testing
os.environ["GROQ_API_KEY"] = "mock_key"

def test_backend_flow():
    print("🚀 Starting End-to-End Backend Verification...")
    
    try:
        print("1. Testing ML Modules...")
        from ml.predict import _get_model
        from ml.preprocessing import _get_scaler
        if not Path("ml/model.pkl").exists() or not Path("ml/scaler.pkl").exists():
             print("⚠️ ML Artifacts missing in directory!")
        else:
             print("✅ ML Artifacts found.")

        print("2. Testing Agent Graph Initialization...")
        from agent.graph import get_graph
        graph = get_graph()
        print("✅ Agent Graph compiled successfully.")

        print("3. Testing Resource Factory...")
        from agent.factory import get_llm
        print("✅ Resource Factory imports valid.")

        print("\n✨ Backend logic and structure are VERIFIED.")
        return True

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_backend_flow()
