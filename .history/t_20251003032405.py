from google import genai
try:
    client = genai.Client()
    # Check if the required method exists
    if hasattr(client, 'generate_content'):
        print("SUCCESS: The 'generate_content' method is available.")
    else:
        print("FAILURE: The 'generate_content' method is still missing.")
except Exception as e:
    print(f"ERROR during verification: {e}")

# Exit the interpreter
exit()