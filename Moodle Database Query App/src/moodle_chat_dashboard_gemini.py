import requests
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import subprocess
import atexit
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")
# DatasetteAPI class
class DatasetteAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def execute_sql(self, database_name, sql_query):
        url = f"{self.base_url}/{database_name}.json"
        params = {"sql": sql_query}
        try:
            response = requests.get(url, params=params, timeout=30)  # Add timeout
            response.raise_for_status()  # Raise HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

# Function to get database schema

def get_schema_with_data(db_path):
    schema = {}
    sample_data = {}

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Fetch table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]

                # Fetch detailed schema information
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                
                # Extract column information: name, type, nullable, and default value
                schema[table_name] = [{
                    "name": column[1],
                    "data type": column[2],
                    "not_null": bool(column[3]),
                    "default_value": column[4]
                } for column in columns]

                # Fetch sample data for the table
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                rows = cursor.fetchall()

                sample_data[table_name] = rows

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        
    return schema, sample_data

# Terminate Datasette process on shutdown
datasette_process = None

def terminate_datasette():
    global datasette_process
    if datasette_process:
        datasette_process.terminate()
        print("Datasette process terminated.")

atexit.register(terminate_datasette)


def full_dashboard(db_path):
    try:
        conn = sqlite3.connect(db_path)

        # Summary Statistics
        st.subheader("Summary Statistics")
        total_users = pd.read_sql("SELECT COUNT(*) AS total_users FROM user;", conn)
        total_courses = pd.read_sql("SELECT COUNT(*) AS total_courses FROM course;", conn)
        total_enrollments = pd.read_sql("SELECT COUNT(*) AS total_enrollments FROM user_enrolments;", conn)
        total_forum_posts = pd.read_sql("SELECT COUNT(*) AS total_posts FROM forum_posts;", conn)
        total_grades = pd.read_sql("SELECT COUNT(*) AS total_grades FROM grade_grades WHERE finalgrade IS NOT NULL;", conn)

        st.write(f"**Total Users:** {total_users}")
        st.write(f"**Total Courses:** {total_courses}")
        st.write(f"**Total Enrollments:** {total_enrollments}")
        st.write(f"**Total Forum Posts:** {total_forum_posts}")
        st.write(f"**Total Grades:** {total_grades}")

        # Distribution of Grades
        st.subheader("Distribution of Grades")
        grades_query = "SELECT finalgrade FROM grade_grades WHERE finalgrade IS NOT NULL;"
        grades = pd.read_sql(grades_query, conn)
        if not grades.empty:
            plt.figure(figsize=(10, 6))
            plt.hist(grades['finalgrade'], bins=20, edgecolor='k', alpha=0.7)
            plt.title("Distribution of Final Grades")
            plt.xlabel("Grade")
            plt.ylabel("Frequency")
            plt.grid()
            st.pyplot(plt)

        # Average Grade per Course
        st.subheader("Average Grade per Course")
        avg_grade_query = """
        SELECT c.fullname AS course, AVG(g.finalgrade) AS average_grade
        FROM grade_grades g
        JOIN grade_items gi ON g.itemid = gi.id
        JOIN course c ON gi.courseid = c.id
        WHERE g.finalgrade IS NOT NULL
        GROUP BY c.fullname
        ORDER BY average_grade DESC;
        """
        course_grades = pd.read_sql(avg_grade_query, conn)
        if not course_grades.empty:
            plt.figure(figsize=(12, 6))
            plt.bar(course_grades['course'], course_grades['average_grade'], color='skyblue')
            plt.xticks(rotation=45, ha='right')
            plt.title("Average Grade per Course")
            plt.xlabel("Course")
            plt.ylabel("Average Grade")
            plt.tight_layout()
            st.pyplot(plt)

        # Course Enrollment Trends
        st.subheader("Course Enrollment Trends")
        enrollments_query = """
        SELECT c.fullname AS course, COUNT(e.userid) AS enrollment_count, e.timecreated
        FROM user_enrolments e
        JOIN course c ON e.enrolid = c.id
        GROUP BY c.fullname, e.timecreated
        ORDER BY e.timecreated;
        """
        enrollments = pd.read_sql(enrollments_query, conn)
        if not enrollments.empty:
            enrollments['timecreated'] = pd.to_datetime(enrollments['timecreated'], unit='s')
            plt.figure(figsize=(12, 6))
            for course in enrollments['course'].unique():
                course_data = enrollments[enrollments['course'] == course]
                plt.plot(course_data['timecreated'], course_data['enrollment_count'], label=course)
            plt.title("Course Enrollment Trends Over Time")
            plt.xlabel("Date")
            plt.ylabel("Enrollment Count")
            plt.legend()
            plt.grid()
            st.pyplot(plt)

        # User Activity Heatmap
        st.subheader("User Activity Heatmap")
        user_activity_query = """
        SELECT userid, courseid, timeaccess
        FROM user_lastaccess
        WHERE timeaccess IS NOT NULL;
        """
        user_activity = pd.read_sql(user_activity_query, conn)
        if not user_activity.empty:
            user_activity['timeaccess'] = pd.to_datetime(user_activity['timeaccess'], unit='s')
            user_activity['hour'] = user_activity['timeaccess'].dt.hour
            user_activity['day'] = user_activity['timeaccess'].dt.day_name()
            heatmap_data = user_activity.pivot_table(index='day', columns='hour', aggfunc='size', fill_value=0)
            plt.figure(figsize=(12, 6))
            sns.heatmap(heatmap_data, cmap='coolwarm', annot=False)
            plt.title("User Activity Heatmap (Day vs Hour)")
            plt.xlabel("Hour of Day")
            plt.ylabel("Day of Week")
            st.pyplot(plt)

        # Forum Posts Per Day
        st.subheader("Forum Posts Per Day")
        forum_posts_query = "SELECT created FROM forum_posts WHERE created IS NOT NULL;"
        forum_posts = pd.read_sql(forum_posts_query, conn)
        if not forum_posts.empty:
            forum_posts['created'] = pd.to_datetime(forum_posts['created'], unit='s')
            forum_posts['date'] = forum_posts['created'].dt.date
            posts_per_day = forum_posts.groupby('date').size()
            plt.figure(figsize=(12, 6))
            posts_per_day.plot(kind='line')
            plt.title("Forum Posts Per Day")
            plt.xlabel("Date")
            plt.ylabel("Number of Posts")
            plt.grid()
            st.pyplot(plt)

        # Grade Comparison Between Courses
        st.subheader("Grade Comparison Between Courses")
        grade_comparison_query = """
        SELECT g.finalgrade, c.fullname AS course
        FROM grade_grades g
        JOIN grade_items gi ON g.itemid = gi.id
        JOIN course c ON gi.courseid = c.id
        WHERE g.finalgrade IS NOT NULL;
        """
        grade_comparison = pd.read_sql(grade_comparison_query, conn)
        if not grade_comparison.empty:
            plt.figure(figsize=(40, 6))
            sns.boxplot(data=grade_comparison, x='course', y='finalgrade')
            plt.title("Grade Comparison Between Courses")
            plt.xlabel("Course")
            plt.ylabel("Final Grade")
            plt.grid()
            st.pyplot(plt)

        # Dynamic Table Viewer
        st.subheader("Explore Data")
        table_names = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)['name'].tolist()
        selected_table = st.selectbox("Select a Table to View", table_names)
        if selected_table:
            table_data = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 100;", conn)
            st.write(f"Displaying {selected_table} data:")
            st.dataframe(table_data)

        conn.close()

    except Exception as e:
        st.error(f"An error occurred: {e}")
        
few_shot_examples = """
User: Who is the top performer in CADE
SQL Query: SELECT DISTINCT T1.firstname, T1.lastname FROM user AS T1 INNER JOIN course_completions AS T2 ON T1.id = T2.userid INNER JOIN course AS T3 ON T2.course = T3.id WHERE T3.shortname LIKE '%CADE%' ORDER BY T2.timecompleted DESC LIMIT 1

User: courses taken by fathima
SQL Query: SELECT DISTINCT T1.fullname FROM course AS T1 INNER JOIN course_completions AS T2 ON T1.id = T2.course INNER JOIN user AS T3 ON T2.userid = T3.id WHERE T3.firstname = 'Fathima'

User: courses taken by taskeen
SQL Query: SELECT DISTINCT c.fullname FROM course c JOIN user_enrolments ue ON c.id = ue.enrolid JOIN user u ON ue.userid = u.id WHERE u.firstname = 'Taskeen';

User: List all courses in which Nikita Shivakumar is enrolled
SQL Query: SELECT DISTINCT mc.fullname FROM user AS mu JOIN user_enrolments AS mue ON mu.id = mue.userid JOIN course AS mc ON mue.enrolid = mc.id WHERE mu.firstname = 'Nikita' AND mu.lastname = 'Shivakumar';

User: What is the average final grade for each courses,give course name?
SQL Query: SELECT AVG(gg.finalgrade), c.fullname FROM grade_grades AS gg JOIN grade_items AS gi ON gg.itemid = gi.id JOIN course AS c ON gi.courseid = c.id WHERE gi.itemtype = 'course' GROUP BY c.fullname;

User: which user have grades highest in course HCM.Provide user name
SQL Query: SELECT DISTINCT T1.firstname, T1.lastname FROM user AS T1 INNER JOIN grade_grades AS T2 ON T1.id = T2.userid INNER JOIN grade_items AS T3 ON T2.itemid = T3.id INNER JOIN course AS T4 ON T3.courseid = T4.id WHERE T4.shortname = 'HCM' ORDER BY T2.finalgrade DESC LIMIT 1

User: Who is the top performer in Maths.Provide name
SQL Query: SELECT DISTINCT T1.firstname, T1.lastname FROM user AS T1 INNER JOIN grade_grades AS T2 ON T1.id = T2.userid INNER JOIN grade_items AS T3 ON T2.itemid = T3.id WHERE T3.itemname LIKE '%Maths%' ORDER BY T2.finalgrade DESC LIMIT 1;

User: compare grades of two courses CADE And Maths
SQL Query: SELECT AVG(CASE WHEN c.shortname = 'CADE' THEN gg.finalgrade ELSE NULL END) AS avg_cade_grade, AVG(CASE WHEN c.shortname = 'Maths' THEN gg.finalgrade ELSE NULL END) AS avg_maths_grade FROM grade_grades AS gg JOIN grade_items AS gi ON gg.itemid = gi.id JOIN course AS c ON gi.courseid = c.id WHERE c.shortname IN ('CADE', 'Maths');

User: course taken by krupa
SQL Query: SELECT DISTINCT c.fullname FROM user AS u JOIN user_enrolments AS ue ON u.id = ue.userid JOIN course AS c ON ue.enrolid = c.id WHERE u.firstname = 'Krupa';

User: show user and the assignment name in which they are enrolled
SQL Query: SELECT DISTINCT T1.firstname, T1.lastname, T3.fullname FROM user AS T1 INNER JOIN assign_submission AS T2 ON T1.id = T2.userid INNER JOIN course AS T3 ON T2.assignment = T3.id
"""

# Main Streamlit app
st.title("Moodle DB Query Application")

# Sidebar navigation
app_mode = st.sidebar.selectbox("Choose the app mode:", ["Chatbot", "Dashboard"])
gemini = GoogleGenerativeAI(model="gemini-2.0-flash-exp", api_key=gemini_api_key, temperature=0.5)

if app_mode == "Chatbot":
    st.header("Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]

    # Upload database
    db_file = os.getenv("DATABASE_FILE")
    

    if "datasette_started" not in st.session_state:
        st.session_state["datasette_started"] = False
    if "db_path" not in st.session_state:
        st.session_state["db_path"] = None
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
    if "conversation_count" not in st.session_state:
        st.session_state["conversation_count"] = 0 
    if db_file and not st.session_state["datasette_started"]:
        # db_path = f"uploaded_{db_file}"
        st.session_state["db_path"] = db_file
        # with open(db_path, "wb") as f:
        #     f.write(db_file.read())

        st.info("Starting Datasette...")
        datasette_process = subprocess.Popen([
            "datasette", "serve", st.session_state["db_path"], "--cors", "--port", "8001"
        ])

        time.sleep(5)
        st.success("Datasette started at http://127.0.0.1:8001/")
        st.session_state["datasette_started"] = True
    # Display chat history
    datasette_api = DatasetteAPI("http://127.0.0.1:8001")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == 'data':
                with st.expander("View Data"):
                    st.write(message["content"])
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your query here..."):
        # Append user input to messages
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # Process user input
        if not gemini_api_key:
            error_message = "Please provide your Google API Key."
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)
        elif st.session_state["db_path"] is None:
            error_message = "Please upload a database file first."
            st.session_state["messages"].append({"role": "assistant", "content": error_message})
            with st.chat_message("assistant"):
                st.markdown(error_message)
        else:
            try:

                schema, sample_data = get_schema_with_data(st.session_state["db_path"])
                conversation_context = st.session_state["memory"].load_memory_variables({})["history"]

                # Retry mechanism
                retries = 3  # Number of retries
                attempt = 0
                while attempt < retries:

                    try:

                        # prompt_template = f"""<s>[INST] 

                        # You are a natural language to SQL query generator. Given the following information:
                        # - **Conversation History:** "{conversation_context}"
                        # - **Database Schema:** "{schema}"
                        # - **few shot example on type of user query:** "{few_shot_examples}"

                        # Generate a SQL query based on the user's query: "{prompt}".

                        # **Requirements for the Generated Query:**
                        # 1. **Key Parameter Identification:** Extract essential parameters such as `user ID`, `date`, or `course`. If missing, infer reasonable values (e.g., today's date for date-related queries or the current user's ID).
                        # 2. **Placeholder Replacement:** Replace any placeholders like `:userid` or `:date` with appropriate inferred values.
                        # 3. **Syntactic Correctness:** Ensure the SQL query follows the correct syntax and adheres to the given database schema for both simple and complex queries.
                        # 4. **Schema-Informed Assumptions:** Infer values based on schema patterns and sample data when not explicitly specified.
                        # 5. **Avoid Duplicates:** Use appropriate clauses such as `DISTINCT` to prevent duplicate results.
                        # 6. **Exact Table Names:** Use the exact table and column names as provided in the schema without any renaming.
                        # 7. **Data Type Validation:** Validate and format values according to schema types (e.g., integers, strings).
                        # 8. **Error Correction:** Correct any spelling errors or incorrect formatting in the query, ensuring alignment with the schema.
                        # 9. **Data Type Handling:** Handle value types correctly, including distinguishing between integers and strings per schema requirements.
                        # 10. **Enrollment Terminology:** Treat the term "enrolled" as indicating user participation or involvement in the specified activity.
                        # 11. **Case Sensitivity:** Handle table and column name case sensitivity appropriately based on the provided schema and sample data.
                        # 12. **Relationship Analysis:** Properly interpret foreign key relationships and references based on the relationship schema.

                        # **Output Requirements:**  

                        # Generate **only** the SQL query without any additional text or explanation.if conversation history not asked by user query

                        # Generate **Conversation History:** done till now if user asked for it in user query and don't give SQL Query For it.

                        # [/INST]"""
                        prompt_template = f"""
                        <s>[INST]

                        You are an advanced natural language to SQL query generator. Based on the following context, generate accurate and schema-compliant SQL queries:

                        - **Conversation History:** "{conversation_context}"  
                        - **Database Schema:** "{schema}"  
                        - **Few-shot Examples:** "{few_shot_examples}"  

                        ### User Query:  
                        "{prompt}"  

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
                        # Generate SQL query using LLM
                        with st.spinner(f"Processing your query... (Attempt {attempt + 1}/{retries})"):
                            response = gemini.invoke(prompt_template)
                            st.write(response)
                            sql_query = response.replace('```sql\n', '').replace('```', '')
                        if "SELECT" in sql_query:
                            query_result = datasette_api.execute_sql(st.session_state["db_path"].split('.')[0], sql_query)
                            if isinstance(query_result, dict) and "rows" in query_result:
                                data = pd.DataFrame(query_result["rows"], columns=query_result["columns"])
                                if not data.empty:
                                    # Convert DataFrame to Markdown for chat persistence
                                    markdown_data = data.to_markdown(index=False)
                                    # Append to session state for persistent chat history
                                    st.session_state["messages"].append({"role": "assistant", "content": "Fetched data displayed below."})
                                    with st.expander("View Data"):
                                        st.write(markdown_data)
                                    st.session_state["messages"].append({"role": "data", "content": f"```\n{markdown_data}\n```"})
                                    prompt1 = f"""
                                    Please convert the table data below into a well-structured, natural language paragraph to answer the query. Format it nicely.
                                    Don't add any extra explanation. Keep it objective.

                                    Query: {prompt}
                                    Data: {data}"""
                                    final_answer = gemini.invoke(prompt1)
                                    st.session_state["memory"].save_context(
                                            {"input": prompt}, {"output": final_answer}
                                    )
                                    st.session_state["messages"].append({"role": "assistant", "content": final_answer})
                                    with st.chat_message("assistant"):
                                        st.write(final_answer)
                                    st.session_state["conversation_count"] += 1
                                    if st.session_state["conversation_count"] >= 9:
                                        # Reset conversation memory and count after 9 interactions
                                        st.session_state["messages"].clear()
                                        st.session_state["memory"].clear()
                                        st.session_state["conversation_count"] = 0
                                        st.write("Conversation memory reset after 9 interactions.")
                                    break
                                else:
                                    no_data_message = f"No data found on attempt"
                                    st.session_state["messages"].append({"role": "assistant", "content": no_data_message})
                                    with st.chat_message("assistant"):
                                        st.markdown(no_data_message)
                                    
                            else:
                                error_message = f"Error executing query"
                                st.session_state["messages"].append({"role": "assistant", "content": error_message})
                                with st.chat_message("assistant"):
                                    st.markdown(error_message)
                                continue  # Exit loop if there's an execution error
                        else:
                            st.session_state["memory"].save_context(
                                    {"input": prompt}, {"output": sql_query}
                            )
                            st.session_state["messages"].append({"role": "assistant", "content": sql_query})
                            with st.chat_message("assistant"):
                                st.write(sql_query)
                            if st.session_state["conversation_count"] >= 9:
                                # Reset conversation memory and count after 9 interactions
                                st.session_state["messages"].clear()
                                st.session_state["memory"].clear()
                                st.session_state["conversation_count"] = 0
                                st.write("Conversation memory reset after 9 interactions.")
                            break
                    except Exception as e:
                        exception_message = f"An error occurred during attempt {attempt + 1}: {e}"
                        st.session_state["messages"].append({"role": "assistant", "content": exception_message})
                        with st.chat_message("assistant"):
                            st.markdown(exception_message)

                    # Increment attempt count and retry
                    attempt += 1
                    time.sleep(2)  # Optional: Add a delay between retries

                # If all retries fail
                if attempt == retries:
                    final_message = "All retry attempts failed. Please refine your query or try again later."
                    st.session_state["messages"].append({"role": "assistant", "content": final_message})
                    with st.chat_message("assistant"):
                        st.markdown(final_message)

            except Exception as e:
                # Handle unexpected errors
                exception_message = f"An error occurred: {e}"
                st.session_state["messages"].append({"role": "assistant", "content": exception_message})
                with st.chat_message("assistant"):
                    st.markdown(exception_message)

# Dashboard section
elif app_mode == "Dashboard":
    st.title("Moodle Dashboard")

    # Upload database
    uploaded_file = os.getenv("DATABASE_FILE")

    if uploaded_file:
        full_dashboard(uploaded_file)
    else:
        st.warning("Please upload a SQLite database file to view the dashboard.")
