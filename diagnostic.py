import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

load_dotenv()

from gsw_memory import GSWProcessor

def main():
    print("Initializing GSWProcessor...")
    try:
        processor = GSWProcessor(
            model_name="gemini/gemini-2.0-flash", 
            enable_coref=False,
            enable_chunking=False,
            enable_context=False,
            enable_spacetime=False
        )
        print("Processor initialized.")
        
        test_text = "John is a doctor. He lives in London."
        print(f"Processing text: {test_text}")
        results = processor.process_documents([test_text])
        print("Success!")
        print(results)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
