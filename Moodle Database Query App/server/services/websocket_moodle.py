import asyncio
import websockets
import json
from moodle_chat_dashboard_gemini import DatasetteAPI, get_schema_with_data, GoogleGenerativeAI, os,few_shot_examples
import pandas as pd
# Initialize Gemini and Datasette API
gemini_api_key = os.getenv("GOOGLE_API_KEY")
gemini = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=gemini_api_key, temperature=0.5)
datasette_api = DatasetteAPI("http://127.0.0.1:8001")
db_path = os.getenv("DATABASE_FILE")

async def handle_websocket(websocket, path):
    async for message in websocket:
        try:
            # Parse the incoming message (user query)
            user_query = json.loads(message)["query"]
            
            # Get schema and sample data
            schema, sample_data = get_schema_with_data(db_path)
            prompt_template = f"""
            <s>[INST]

            You are an advanced natural language to SQL query generator. Based on the following context, generate accurate and schema-compliant SQL queries:
            - **Database Schema:** "{schema}"  
            - **Few-shot Examples:** "{few_shot_examples}"  

            ### User Query:  
            "{user_query}"  

            ### Requirements for Generated SQL Queries:
            1. **Key Parameter Identification:** Extract essential parameters like `user ID`, `date`, or `course`. If missing, infer reasonable values such as today's date for date-related queries or the current user's ID.  
            2. **Placeholder Replacement:** Replace placeholders (e.g., `:userid`, `:date`) with inferred values where possible.  
            3. **Schema Compliance:** Ensure the query strictly follows the provided database schema, adhering to exact table and column names.  
            4. **Syntactic Accuracy:** Maintain syntactic correctness for both simple and complex queries.  
            5. **Distinct Results:** Use `DISTINCT` or other appropriate clauses to avoid duplicate results.  
            6. **Data Type Handling:** Validate and format values based on schema types, distinguishing between integers and strings as necessary.  
            7. **Foreign Key Relationships:** Properly interpret and apply schema-defined relationships between tables.  
            8. **Enrollment Context:** Treat "enrolled" as user participation in a specified activity or course.  
            9. **Error Handling:** Correct spelling errors or incorrect formatting while maintaining schema alignment.  
            10. **Case Sensitivity:** Handle table and column name case sensitivity per the provided schema.  
            11. **Schema-Informed Assumptions:** Infer missing information based on schema patterns when needed.

            ### Output Rules:
            - **SQL Query Only:** If the user query does not request conversation history, generate **only the SQL query** without explanations.  
            - **Conversation History:** If the user explicitly requests conversation history, provide only the conversation context without generating an SQL query.  

            [/INST]
            """
            sql_query = gemini.invoke(prompt_template).replace('```sql\n', '').replace('```', '')
            
            # Execute the SQL query using Datasette API
            query_result = datasette_api.execute_sql(db_path.split('.')[0], sql_query)
            
            # Prepare the response
            if isinstance(query_result, dict) and "rows" in query_result:
                data = pd.DataFrame(query_result["rows"], columns=query_result["columns"])
                response = {
                    "status": "success",
                    "data": data.to_dict(orient="records")
                }
            else:
                response = {
                    "status": "error",
                    "message": "No data found or query execution failed."
                }
            
            # Send the response back to the client
            await websocket.send(json.dumps(response))
        
        except Exception as e:
            # Handle errors
            error_response = {
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }
            await websocket.send(json.dumps(error_response))

# Start the WebSocket server
async def start_websocket_server():
    async with websockets.serve(handle_websocket, "localhost", 8765):
        await asyncio.Future()  # Run forever

# Run the WebSocket server
if __name__ == "__main__":
    asyncio.run(start_websocket_server())