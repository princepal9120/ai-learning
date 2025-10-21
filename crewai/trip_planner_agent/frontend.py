from travelAgents import guide_expert, location_expert, planner_expert
from travelTasks import location_task, guide_task, planner_task
from crewai import Crew, Process
import streamlit as st

# Streamlit App Title
st.title("🌍 AI-Powered Trip Planner")

st.markdown("""
💡 **Plan your next trip with AI!**  
Enter your travel details below, and our AI-powered travel assistant will create a personalized itinerary including:
 Best places to visit 🎡   Accommodation & budget planning 💰
 Local food recommendations 🍕   Transportation & visa details 🚆
""")

# User Inputs
from_city = st.text_input("🏡 From City", "India")
destination_city = st.text_input("✈️ Destination City", "Rome")
date_from = st.date_input("📅 Departure Date")
date_to = st.date_input("📅 Return Date")
interests = st.text_area("🎯 Your Interests (e.g., sightseeing, food, adventure)", "sightseeing and good food")

# Button to run CrewAI
if st.button("🚀 Generate Travel Plan"):
    if not from_city or not destination_city or not date_from or not date_to or not interests:
        st.error("⚠️ Please fill in all fields before generating your travel plan.")
    else:
        st.write("⏳ AI is preparing your personalized travel itinerary... Please wait.")

        # Initialize Tasks
        loc_task = location_task(location_expert, from_city, destination_city, date_from, date_to)
        guid_task = guide_task(guide_expert, destination_city, interests, date_from, date_to)
        plan_task = planner_task([loc_task, guid_task], planner_expert, destination_city, interests, date_from, date_to)

        # Define Crew
        crew = Crew(
            agents=[location_expert, guide_expert, planner_expert],
            tasks=[loc_task, guid_task, plan_task],
            process=Process.sequential,
            full_output=True,
            verbose=True,
        )

        # Run Crew AI
        result = crew.kickoff()

        # Display Results
        st.subheader("✅ Your AI-Powered Travel Plan")
        st.markdown(result)


        # Ensure result is a string
        travel_plan_text = str(result)  # ✅ Convert CrewOutput to string

        st.download_button(
            label="📥 Download Travel Plan",
            data=travel_plan_text,  # ✅ Now passing a valid string
            file_name=f"Travel_Plan_{destination_city}.txt",
            mime="text/plain"
        )