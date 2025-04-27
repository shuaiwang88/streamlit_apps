import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import streamlit as st
import matplotlib.pyplot as plt
import time

# Page config
st.set_page_config(page_title="Classroom Scheduling Optimization", page_icon="🎓")

# ================== Streamlit App ==================
st.title("🎓 Classroom Scheduling Optimization")
st.markdown("""
### Optimize course scheduling and classroom assignments
Maximize resource utilization and minimize conflicts in academic scheduling.
""")

# ================== Possible Extensions (Collapsible Section) ==================
with st.expander("🚀 **Industry Applications (Click to Expand)**"):
    st.markdown("""
    **Education optimization applications:**  
    - **Course Scheduling**: Assign courses to rooms and time slots to minimize conflicts.
    - **Faculty Workload Balancing**: Distribute teaching assignments fairly among faculty.
    - **Student Scheduling**: Create optimal student schedules to minimize gaps between classes.
    - **Exam Scheduling**: Schedule exams to avoid student conflicts and distribute workload.
    - **Resource Allocation**: Optimize the use of classrooms, labs, and specialized equipment.
    """)

# ================== Input Parameters ==================
st.sidebar.header("Scheduling Parameters")

# Basic parameters
num_courses = st.sidebar.slider("Number of Courses", 10, 50, 25)
num_classrooms = st.sidebar.slider("Number of Classrooms", 5, 20, 10)
num_time_slots = st.sidebar.slider("Number of Time Slots per Day", 4, 12, 8)
num_days = st.sidebar.slider("Number of Days in Schedule", 1, 5, 5)
num_professors = st.sidebar.slider("Number of Professors", 5, 20, 15)

# Complexity parameters
min_course_sessions = st.sidebar.slider("Min Sessions per Course", 1, 3, 2)
max_course_sessions = st.sidebar.slider("Max Sessions per Course", 2, 5, 3)
avg_class_size = st.sidebar.slider("Average Class Size", 15, 60, 30)
classroom_specialization = st.sidebar.slider("Classroom Specialization Level", 0, 100, 40)

# Generate random data
@st.cache_data
def generate_data(seed=42):
    np.random.seed(seed)
    
    # Courses, classrooms, time slots, and professors
    courses = [f"Course_{i+1}" for i in range(num_courses)]
    classrooms = [f"Room_{j+1}" for j in range(num_classrooms)]
    days = [f"Day_{d+1}" for d in range(num_days)]
    time_slots = [f"Slot_{t+1}" for t in range(num_time_slots)]
    professors = [f"Prof_{p+1}" for p in range(num_professors)]
    
    # Time periods (combinations of days and time slots)
    periods = [(d, t) for d in days for t in time_slots]
    
    # Course attributes
    course_sessions = {c: np.random.randint(min_course_sessions, max_course_sessions + 1) 
                      for c in courses}
    
    # Each course has approximately 20% of available time slots marked as preferred
    course_preferred_periods = {c: np.random.choice(
        periods, 
        size=int(0.2 * len(periods)), 
        replace=False
    ) for c in courses}
    
    # Course enrollment (number of students)
    enrollment = {c: np.random.randint(avg_class_size * 0.5, avg_class_size * 1.5) for c in courses}
    
    # Classroom capacity
    capacity = {r: np.random.randint(20, 100) for r in classrooms}
    
    # Classroom specialization (e.g., labs, lecture halls)
    # We'll use 4 types: 0 = standard, 1 = computer lab, 2 = science lab, 3 = seminar room
    room_type = {r: np.random.randint(0, 4) for r in classrooms}
    
    # Course room type requirements
    course_room_type = {c: np.random.randint(0, 4) for c in courses}
    
    # Course compatibility with room types (1 = compatible)
    room_compatibility = {}
    for c in courses:
        for r in classrooms:
            # Higher classroom_specialization = stricter compatibility requirements
            if classroom_specialization <= 20:
                # Low specialization: All rooms work for all courses
                compatibility = 1
            elif classroom_specialization <= 60:
                # Medium specialization: Some requirements
                if course_room_type[c] == 0 or room_type[r] == 0 or course_room_type[c] == room_type[r]:
                    compatibility = 1
                else:
                    compatibility = 0
            else:
                # High specialization: Strict requirements
                if course_room_type[c] == room_type[r]:
                    compatibility = 1
                else:
                    compatibility = 0
            room_compatibility[(c, r)] = compatibility
    
    # Professor assignments to courses (each professor teaches ~3 courses)
    prof_course_assignments = {}
    for p in professors:
        # Assign each professor to about 3 courses
        assigned_courses = np.random.choice(courses, size=min(3, len(courses)), replace=False)
        for c in courses:
            prof_course_assignments[(p, c)] = 1 if c in assigned_courses else 0
    
    # Professor availability (professors available ~80% of periods)
    prof_availability = {}
    for p in professors:
        # Generate random availability (1 = available)
        for period in periods:
            prof_availability[(p, period)] = np.random.choice([0, 1], p=[0.2, 0.8])
    
    return (courses, classrooms, days, time_slots, periods, course_sessions, 
            course_preferred_periods, enrollment, capacity, room_type, course_room_type,
            room_compatibility, prof_course_assignments, prof_availability)

# Generate data
(courses, classrooms, days, time_slots, periods, course_sessions, 
 course_preferred_periods, enrollment, capacity, room_type, course_room_type,
 room_compatibility, prof_course_assignments, prof_availability) = generate_data()

# Map room types to readable names
room_type_names = {0: "Standard", 1: "Computer Lab", 2: "Science Lab", 3: "Seminar Room"}

# ================== Pyomo Model ==================
def solve_scheduling_optimization():
    # Create model
    model = pyo.ConcreteModel()
    
    # Sets
    model.Courses = pyo.Set(initialize=courses)
    model.Classrooms = pyo.Set(initialize=classrooms)
    model.Days = pyo.Set(initialize=days)
    model.TimeSlots = pyo.Set(initialize=time_slots)
    model.Periods = pyo.Set(initialize=periods)
    model.Professors = pyo.Set(initialize=professors)
    
    # Parameters
    model.CourseSessions = pyo.Param(model.Courses, initialize=course_sessions)
    model.Enrollment = pyo.Param(model.Courses, initialize=enrollment)
    model.Capacity = pyo.Param(model.Classrooms, initialize=capacity)
    model.RoomCompatibility = pyo.Param(model.Courses, model.Classrooms, initialize=room_compatibility)
    model.ProfCourseAssignments = pyo.Param(model.Professors, model.Courses, initialize=prof_course_assignments)
    model.ProfAvailability = pyo.Param(model.Professors, model.Periods, initialize=prof_availability)
    
    # Course preference for time periods (1 = preferred)
    course_period_preference = {}
    for c in courses:
        for period in periods:
            course_period_preference[(c, period)] = 1 if period in course_preferred_periods[c] else 0
    model.CoursePeriodPreference = pyo.Param(model.Courses, model.Periods, initialize=course_period_preference)
    
    # Decision Variables
    # Schedule[c,r,d,t] = 1 if course c is scheduled in room r on day d, time slot t
    model.Schedule = pyo.Var(model.Courses, model.Classrooms, model.Days, model.TimeSlots, domain=pyo.Binary)
    
    # Objective function: maximize preferred periods and room compatibility
    def objective_rule(m):
        # Term 1: Maximize scheduling in preferred periods
        preferred_periods_score = sum(
            m.CoursePeriodPreference[c, (d, t)] * m.Schedule[c, r, d, t]
            for c in m.Courses for r in m.Classrooms for d in m.Days for t in m.TimeSlots
        )
        
        # Term 2: Maximize room compatibility
        room_compatibility_score = sum(
            m.RoomCompatibility[c, r] * m.Schedule[c, r, d, t]
            for c in m.Courses for r in m.Classrooms for d in m.Days for t in m.TimeSlots
        )
        
        # Term 3: Bonus for scheduling all required sessions
        sessions_scheduled = {}
        for c in m.Courses:
            sessions_scheduled[c] = sum(
                m.Schedule[c, r, d, t] 
                for r in m.Classrooms for d in m.Days for t in m.TimeSlots
            )
        
        sessions_met_score = sum(
            100 if sessions_scheduled[c] >= m.CourseSessions[c] else 0
            for c in m.Courses
        )
        
        return preferred_periods_score + room_compatibility_score + sessions_met_score
    
    model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # Constraints
    
    # C1: Course can be scheduled at most once per time period
    def one_room_per_period_rule(m, c, d, t):
        return sum(m.Schedule[c, r, d, t] for r in m.Classrooms) <= 1
    
    model.OneRoomPerPeriod = pyo.Constraint(model.Courses, model.Days, model.TimeSlots, 
                                           rule=one_room_per_period_rule)
    
    # C2: Room can host at most one course per time period
    def one_course_per_room_rule(m, r, d, t):
        return sum(m.Schedule[c, r, d, t] for c in m.Courses) <= 1
    
    model.OneCoursePerRoom = pyo.Constraint(model.Classrooms, model.Days, model.TimeSlots, 
                                           rule=one_course_per_room_rule)
    
    # C3: Course sessions constraint (each course must be scheduled exactly for its required number of sessions)
    def course_sessions_rule(m, c):
        return sum(m.Schedule[c, r, d, t] for r in m.Classrooms for d in m.Days 
                  for t in m.TimeSlots) == m.CourseSessions[c]
    
    model.CourseSessionsConstraint = pyo.Constraint(model.Courses, rule=course_sessions_rule)
    
    # C4: Room capacity constraint
    def room_capacity_rule(m, c, r, d, t):
        return m.Enrollment[c] * m.Schedule[c, r, d, t] <= m.Capacity[r]
    
    model.RoomCapacityConstraint = pyo.Constraint(model.Courses, model.Classrooms, model.Days, 
                                                 model.TimeSlots, rule=room_capacity_rule)
    
    # C5: Professor can't teach more than one course at the same time
    def prof_schedule_rule(m, p, d, t):
        return sum(m.ProfCourseAssignments[p, c] * m.Schedule[c, r, d, t] 
                  for c in m.Courses for r in m.Classrooms) <= 1
    
    model.ProfScheduleConstraint = pyo.Constraint(model.Professors, model.Days, model.TimeSlots, 
                                                 rule=prof_schedule_rule)
    
    # C6: Courses must be scheduled when the professor is available
    def prof_availability_rule(m, p, c, d, t):
        if m.ProfCourseAssignments[p, c] == 1:
            # If professor p teaches course c, then course c can only be scheduled 
            # in periods where professor p is available
            return sum(m.Schedule[c, r, d, t] for r in m.Classrooms) <= m.ProfAvailability[p, (d, t)]
        else:
            # If professor p doesn't teach course c, no constraint needed
            return pyo.Constraint.Skip
    
    model.ProfAvailabilityConstraint = pyo.Constraint(model.Professors, model.Courses, model.Days, 
                                                     model.TimeSlots, rule=prof_availability_rule)
    
    # C7: Room compatibility constraint
    def compatibility_rule(m, c, r, d, t):
        return m.Schedule[c, r, d, t] <= m.RoomCompatibility[c, r]
    
    model.CompatibilityConstraint = pyo.Constraint(model.Courses, model.Classrooms, model.Days, 
                                                  model.TimeSlots, rule=compatibility_rule)
    
    # Solve the model
    solver = SolverFactory('appsi_highs')
    start_time = time.time()
    results = solver.solve(model)
    solve_time = time.time() - start_time
    
    return model, results, solve_time

# ================== Run Optimization ==================
if st.button("Run Classroom Scheduling Optimization"):
    with st.spinner("Optimizing classroom schedule..."):
        model, results, solve_time = solve_scheduling_optimization()
    
    if results.solver.status == pyo.SolverStatus.ok:
        st.success(f"✅ Optimization complete in {solve_time:.2f} seconds!")
        
        # Extract schedule from solution
        schedule_data = []
        for c in courses:
            for r in classrooms:
                for d in days:
                    for t in time_slots:
                        if pyo.value(model.Schedule[c, r, d, t]) > 0.5:
                            # Find professor teaching this course
                            professor = next((p for p in professors if prof_course_assignments[(p, c)] == 1), None)
                            
                            schedule_data.append({
                                "Course": c,
                                "Room": r,
                                "Day": d,
                                "Time Slot": t,
                                "Students": enrollment[c],
                                "Room Capacity": capacity[r],
                                "Room Type": room_type_names[room_type[r]],
                                "Professor": professor
                            })
        
        # Create schedule dataframe
        if schedule_data:
            schedule_df = pd.DataFrame(schedule_data)
            
            # Calculate metrics
            total_sessions = len(schedule_df)
            sessions_in_preferred_periods = sum(1 for _, row in schedule_df.iterrows() 
                                              if (row["Day"], row["Time Slot"]) in course_preferred_periods[row["Course"]])
            preferred_period_pct = sessions_in_preferred_periods / total_sessions * 100 if total_sessions > 0 else 0
            
            avg_capacity_utilization = schedule_df["Students"].sum() / schedule_df["Room Capacity"].sum() * 100
            
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Scheduled Sessions", f"{total_sessions}")
            col2.metric("Preferred Period Usage", f"{preferred_period_pct:.1f}%")
            col3.metric("Capacity Utilization", f"{avg_capacity_utilization:.1f}%")
            
            # Display schedule in different views
            tab1, tab2, tab3 = st.tabs(["Room Schedule", "Course Schedule", "Visualizations"])
            
            with tab1:
                # Group by room and display timetable for each room
                st.subheader("Room Schedules")
                
                # Let user select a room to view
                selected_room = st.selectbox("Select Room to View Schedule", classrooms)
                
                # Filter for selected room
                room_schedule = schedule_df[schedule_df["Room"] == selected_room]
                
                # Create a timetable for the selected room
                timetable = pd.DataFrame(index=time_slots, columns=days)
                
                # Fill in the timetable
                for _, row in room_schedule.iterrows():
                    timetable.at[row["Time Slot"], row["Day"]] = f"{row['Course']} ({row['Professor']})"
                
                # Display the timetable
                st.write(f"**Schedule for {selected_room} (Capacity: {capacity[selected_room]}, Type: {room_type_names[room_type[selected_room]]})**")
                st.dataframe(timetable.fillna(""), use_container_width=True)
            
            with tab2:
                # Group by course and display when and where each course is scheduled
                st.subheader("Course Schedules")
                
                # Let user select a course to view
                selected_course = st.selectbox("Select Course to View Schedule", courses)
                
                # Filter for selected course
                course_schedule = schedule_df[schedule_df["Course"] == selected_course]
                
                # Display the schedule for this course
                if not course_schedule.empty:
                    st.write(f"**Schedule for {selected_course}**")
                    st.write(f"Enrollment: {enrollment[selected_course]} students")
                    st.write(f"Professor: {course_schedule['Professor'].iloc[0]}")
                    st.write(f"Required Room Type: {room_type_names[course_room_type[selected_course]]}")
                    st.write(f"Sessions Required: {course_sessions[selected_course]}")
                    
                    # Create a simplified view of when and where this course is taught
                    sessions_table = course_schedule[["Day", "Time Slot", "Room"]].sort_values(["Day", "Time Slot"])
                    st.dataframe(sessions_table, use_container_width=True)
                else:
                    st.write("No schedule found for this course.")
            
            with tab3:
                # Visualizations of the schedule
                st.subheader("Schedule Visualizations")
                
                # 1. Room usage chart
                room_usage = schedule_df["Room"].value_counts()
                
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                ax1.bar(room_usage.index, room_usage.values)
                ax1.set_xlabel("Room")
                ax1.set_ylabel("Number of Sessions")
                ax1.set_title("Room Usage Frequency")
                plt.xticks(rotation=45)
                ax1.grid(axis="y", linestyle="--", alpha=0.7)
                st.pyplot(fig1)
                
                # 2. Time slot popularity
                timeslot_usage = schedule_df["Time Slot"].value_counts().sort_index()
                
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                ax2.bar(timeslot_usage.index, timeslot_usage.values)
                ax2.set_xlabel("Time Slot")
                ax2.set_ylabel("Number of Sessions")
                ax2.set_title("Time Slot Usage")
                ax2.grid(axis="y", linestyle="--", alpha=0.7)
                st.pyplot(fig2)
                
                # 3. Day of week distribution
                day_usage = schedule_df["Day"].value_counts().sort_index()
                
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.bar(day_usage.index, day_usage.values)
                ax3.set_xlabel("Day")
                ax3.set_ylabel("Number of Sessions")
                ax3.set_title("Sessions per Day")
                ax3.grid(axis="y", linestyle="--", alpha=0.7)
                st.pyplot(fig3)
                
            # Full schedule view
            with st.expander("View Full Schedule"):
                st.dataframe(schedule_df, use_container_width=True)
        else:
            st.warning("No feasible schedule found. Try relaxing constraints.")
    else:
        st.error("Optimization failed to find a solution.")

# ================== Data Preview ==================
with st.expander("📊 Preview Input Data"):
    data_tab1, data_tab2, data_tab3 = st.tabs(["Courses", "Classrooms", "Professors"])
    
    with data_tab1:
        # Course data
        course_df = pd.DataFrame({
            "Course": courses,
            "Required Sessions": [course_sessions[c] for c in courses],
            "Enrollment": [enrollment[c] for c in courses],
            "Room Type Needed": [room_type_names[course_room_type[c]] for c in courses],
        })
        st.write("**Course Information**")
        st.dataframe(course_df)
    
    with data_tab2:
        # Classroom data
        classroom_df = pd.DataFrame({
            "Room": classrooms,
            "Capacity": [capacity[r] for r in classrooms],
            "Room Type": [room_type_names[room_type[r]] for r in classrooms]
        })
        st.write("**Classroom Information**")
        st.dataframe(classroom_df)
    
    with data_tab3:
        # Professor teaching assignments
        prof_assignments = []
        for p in professors:
            taught_courses = [c for c in courses if prof_course_assignments[(p, c)] == 1]
            prof_assignments.append({
                "Professor": p,
                "Courses Taught": ", ".join(taught_courses),
                "Number of Courses": len(taught_courses)
            })
        
        prof_df = pd.DataFrame(prof_assignments)
        st.write("**Professor Teaching Assignments**")
        st.dataframe(prof_df)