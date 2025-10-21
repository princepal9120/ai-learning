from crewai import Task

def location_task(agent, from_city, destination_city, date_from, date_to):
    return Task(
        description=f"""
        In French: Provide travel-related information including accommodations, cost of living,
        visa requirements, transportation, weather, and local events.

        Traveling from: {from_city}
        Destination: {destination_city}
        Arrival Date: {date_from}
        Departure Date: {date_to}

        Respond in FRENCH if the destination is in a French-speaking country.
        """,
        expected_output="A detailed markdown report with relevant travel data.",
        agent=agent,
        output_file='city_report.md',
    )

def guide_task(agent, destination_city, interests, date_from, date_to):
    return Task(
        description=f"""
        Provide a travel guide with attractions, food recommendations, and events.
        Tailor recommendations based on user interests: {interests}.
        
        Destination: {destination_city}
        Arrival Date: {date_from}
        Departure Date: {date_to}
        """,
        expected_output="A markdown itinerary including attractions, food, and activities.",
        agent=agent,
        output_file='guide_report.md',
    )

def planner_task(context, agent, destination_city, interests, date_from, date_to):
    return Task(
        description=f"""
        Combine information into a well-structured itinerary. Include:
        - City introduction (4 paragraphs)
        - Daily travel plan with time allocations
        - Expenses and tips

        Destination: {destination_city}
        Interests: {interests}
        Arrival: {date_from}
        Departure: {date_to}
        """,
        expected_output="A structured markdown travel itinerary.",
        context=context,
        agent=agent,
        output_file='travel_plan.md',
    )