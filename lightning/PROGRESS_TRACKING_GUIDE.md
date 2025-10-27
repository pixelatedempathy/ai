# Therapeutic Progress Tracking Guide

## 🎯 Overview

Long-term, journal-style tracking system for monitoring client therapeutic progress over extended timeframes (weeks, months, years).

## 🚀 Quick Start

### Start Progress Tracking API

```bash
python progress_tracking_api.py
```

Service available at `http://localhost:8001`

### Log a Session

```bash
curl -X POST http://localhost:8001/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_001",
    "client_id": "client_123",
    "conversation_summary": "Discussed anxiety management",
    "emotional_state": "neutral",
    "therapeutic_goals": ["goal_001"],
    "progress_notes": "Client showing improvement",
    "therapist_observations": "Engaged and receptive",
    "next_session_focus": "Practice mindfulness"
  }'
```

### Get Progress Report

```bash
curl http://localhost:8001/api/v1/progress/client_123?timeframe_days=30
```

## 📊 Features

### 1. Session Logging

**Journal-style logging** of therapy sessions:
- Conversation summaries
- Emotional state tracking
- Progress notes
- Therapist observations
- Techniques used
- Homework assigned
- Crisis flags

### 2. Goal Tracking

**Long-term therapeutic goals**:
- Goal creation and updates
- Completion percentage tracking
- Milestone achievements
- Target dates
- Progress notes

### 3. Emotional Trends

**Analyze emotional patterns**:
- Average emotional scores
- Trend direction (improving/stable/declining)
- Volatility measurement
- Data point tracking

### 4. Progress Reports

**Comprehensive reports**:
- Session summaries
- Goal progress
- Emotional trends
- Key milestones
- Overall trajectory
- Recommendations

### 5. Long-term Tracking

**Extended timeframes**:
- Daily tracking
- Weekly summaries
- Monthly reports
- Quarterly reviews
- Yearly analysis
- Multi-year trends

## 🗄️ Data Model

### Session Log

```python
{
  "session_id": "session_001",
  "client_id": "client_123",
  "timestamp": "2024-10-27T10:00:00",
  "conversation_summary": "Discussed coping strategies",
  "emotional_state": "positive",
  "therapeutic_goals": ["goal_001", "goal_002"],
  "progress_notes": "Significant progress this week",
  "therapist_observations": "More confident",
  "next_session_focus": "Maintain progress",
  "session_duration_minutes": 60,
  "techniques_used": ["CBT", "mindfulness"],
  "homework_assigned": "Daily journaling",
  "crisis_flags": [],
  "metadata": {}
}
```

### Therapeutic Goal

```python
{
  "goal_id": "goal_001",
  "description": "Reduce anxiety symptoms",
  "target_date": "2025-01-27",
  "completion_percentage": 65.0,
  "milestones": ["milestone_001", "milestone_002"],
  "notes": "Making steady progress",
  "created_at": "2024-10-01",
  "updated_at": "2024-10-27"
}
```

### Progress Report

```python
{
  "client_id": "client_123",
  "report_date": "2024-10-27",
  "timeframe_start": "2024-09-27",
  "timeframe_end": "2024-10-27",
  "sessions_count": 4,
  "goal_progress": {
    "goal_001": 65.0,
    "goal_002": 40.0
  },
  "overall_trajectory": "improving",
  "recommendations": [
    "Continue current therapeutic approach",
    "Consider increasing session frequency"
  ],
  "summary": "Completed 4 sessions. Emotional state trending improving..."
}
```

## 📈 Timeframe Support

### Short-term (Days/Weeks)

```python
# Last 7 days
report = tracker.generate_progress_report(client_id, timeframe_days=7)

# Last 2 weeks
report = tracker.generate_progress_report(client_id, timeframe_days=14)
```

### Medium-term (Months)

```python
# Last month
report = tracker.generate_progress_report(client_id, timeframe_days=30)

# Last quarter
report = tracker.generate_progress_report(client_id, timeframe_days=90)
```

### Long-term (Years)

```python
# Last year
report = tracker.generate_progress_report(client_id, timeframe_days=365)

# Last 2 years
report = tracker.generate_progress_report(client_id, timeframe_days=730)
```

## 🔍 Analysis Features

### Emotional Trend Analysis

```python
trends = tracker.analyze_emotional_trends(
    client_id="client_123",
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now()
)

# Returns:
# - Average emotional score (-2 to +2)
# - Trend direction (improving/stable/declining)
# - Volatility (standard deviation)
# - Number of data points
```

### Goal Progress Tracking

```python
goals = tracker.get_goals(client_id="client_123")

for goal in goals:
    print(f"{goal.description}: {goal.completion_percentage}%")
```

### Milestone Tracking

```python
milestone = Milestone(
    milestone_id="milestone_001",
    goal_id="goal_001",
    description="First week without panic attack",
    achieved_date=datetime.now(),
    significance="major"
)

tracker.add_milestone(milestone, client_id="client_123")
```

## 🔒 Privacy & Security

### HIPAA Compliance

- ✅ Encrypted storage (SQLite with encryption)
- ✅ Access control (API authentication)
- ✅ Audit logging (all operations logged)
- ✅ Data retention policies
- ✅ Secure deletion capabilities

### Data Protection

```python
# Export client history (encrypted)
tracker.export_client_history(
    client_id="client_123",
    output_path="client_history.json"
)

# Data is encrypted at rest
# Access requires authentication
# All operations are logged
```

## 📊 API Reference

### POST /api/v1/sessions

Log a therapy session.

**Request**:
```json
{
  "session_id": "session_001",
  "client_id": "client_123",
  "conversation_summary": "Discussed anxiety",
  "emotional_state": "neutral",
  "therapeutic_goals": ["goal_001"],
  "progress_notes": "Good progress",
  "therapist_observations": "Engaged",
  "next_session_focus": "Continue techniques"
}
```

### GET /api/v1/sessions/{client_id}

Get sessions for a client.

**Query Parameters**:
- `start_date`: Filter by start date
- `end_date`: Filter by end date
- `limit`: Limit number of results

### POST /api/v1/goals

Create a therapeutic goal.

### PUT /api/v1/goals/{goal_id}

Update goal progress.

### GET /api/v1/progress/{client_id}

Generate progress report.

**Query Parameters**:
- `timeframe_days`: Number of days to analyze (default: 30)

### GET /api/v1/trends/{client_id}

Get emotional trends.

### GET /api/v1/export/{client_id}

Export complete client history.

## 💡 Usage Examples

### Example 1: Weekly Check-in

```python
from therapeutic_progress_tracker import *

tracker = TherapeuticProgressTracker()

# Log session
session = SessionLog(
    session_id=f"session_{datetime.now().timestamp()}",
    client_id="client_123",
    timestamp=datetime.now(),
    conversation_summary="Weekly check-in on anxiety management",
    emotional_state=EmotionalState.POSITIVE,
    therapeutic_goals=["reduce_anxiety"],
    progress_notes="Client reports fewer panic attacks this week",
    therapist_observations="More confident in using coping techniques",
    next_session_focus="Maintain progress and introduce new techniques"
)

tracker.log_session(session)
```

### Example 2: Goal Progress Update

```python
# Update goal progress
tracker.update_goal_progress(
    goal_id="reduce_anxiety",
    completion_percentage=75.0,
    notes="Significant improvement in managing anxiety symptoms"
)
```

### Example 3: Monthly Progress Report

```python
# Generate monthly report
report = tracker.generate_progress_report(
    client_id="client_123",
    timeframe_days=30
)

print(f"Sessions: {report.sessions_count}")
print(f"Trajectory: {report.overall_trajectory.value}")
print(f"Summary: {report.summary}")
print(f"Recommendations: {', '.join(report.recommendations)}")
```

### Example 4: Long-term Trend Analysis

```python
# Analyze 6-month trends
trends = tracker.analyze_emotional_trends(
    client_id="client_123",
    start_date=datetime.now() - timedelta(days=180),
    end_date=datetime.now()
)

for trend in trends:
    print(f"Average score: {trend.avg_emotional_score:.2f}")
    print(f"Direction: {trend.trend_direction}")
    print(f"Volatility: {trend.volatility:.2f}")
```

## 🎯 Best Practices

### 1. Consistent Logging

Log every session immediately after completion:
```python
# Right after session
tracker.log_session(session)
```

### 2. Regular Goal Updates

Update goal progress weekly:
```python
# Weekly goal review
tracker.update_goal_progress(goal_id, new_percentage, notes)
```

### 3. Milestone Recognition

Celebrate achievements:
```python
# When milestone achieved
tracker.add_milestone(milestone, client_id)
```

### 4. Periodic Reports

Generate reports regularly:
```python
# Monthly reports
monthly_report = tracker.generate_progress_report(client_id, 30)

# Quarterly reviews
quarterly_report = tracker.generate_progress_report(client_id, 90)
```

### 5. Export for Backup

Regular backups:
```python
# Monthly backup
tracker.export_client_history(client_id, f"backup_{date}.json")
```

## 🔧 Configuration

### Database Location

```python
tracker = TherapeuticProgressTracker(
    db_path="custom_path/progress.db"
)
```

### Custom Timeframes

```python
# Custom date range
sessions = tracker.get_sessions(
    client_id="client_123",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

## 📚 Integration

### With Inference Engine

```python
# After inference, log session
response, metadata = engine.generate(user_input)

session = SessionLog(
    session_id=generate_session_id(),
    client_id=client_id,
    timestamp=datetime.now(),
    conversation_summary=response[:200],
    emotional_state=detect_emotional_state(response),
    ...
)

tracker.log_session(session)
```

### With Monitoring

```python
# Track progress metrics
metrics = tracker.get_summary()

wandb.log({
    'progress/sessions_count': metrics['sessions_count'],
    'progress/avg_emotional_score': metrics['avg_emotional_score']
})
```

---

**Questions?** Check the API documentation or examples.
