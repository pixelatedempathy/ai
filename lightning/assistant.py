import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import re

class PersonalAssistant:
    def __init__(self, data_file: str = "assistant_data.json"):
        self.data_file = data_file
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load assistant data from file"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        
        # Default data structure
        return {
            "tasks": [],
            "preferences": {},
            "schedule": {},
            "reminders": []
        }
    
    def _save_data(self):
        """Save assistant data to file"""
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def add_task(self, task: str, priority: str = "medium", due_date: Optional[str] = None) -> str:
        """Add a new task"""
        task_id = len(self.data["tasks"]) + 1
        new_task = {
            "id": task_id,
            "task": task,
            "priority": priority.lower(),
            "due_date": due_date,
            "completed": False,
            "created_at": datetime.now().isoformat()
        }
        
        self.data["tasks"].append(new_task)
        self._save_data()
        return f"✅ Added task #{task_id}: {task}"
    
    def complete_task(self, task_id: int) -> str:
        """Mark a task as completed"""
        for task in self.data["tasks"]:
            if task["id"] == task_id:
                task["completed"] = True
                task["completed_at"] = datetime.now().isoformat()
                self._save_data()
                return f"🎉 Completed task #{task_id}: {task['task']}"
        return f"❌ Task #{task_id} not found"
    
    def list_tasks(self, show_completed: bool = False) -> str:
        """List all tasks"""
        tasks = [t for t in self.data["tasks"] if show_completed or not t["completed"]]
        
        if not tasks:
            return "📝 No tasks found"
        
        result = "📋 **Your Tasks:**\n"
        for task in sorted(tasks, key=lambda x: (x["completed"], x["priority"] != "high")):
            status = "✅" if task["completed"] else "⏳"
            priority_emoji = {"high": "🔥", "medium": "📌", "low": "💡"}.get(task["priority"], "📌")
            due_info = f" (Due: {task['due_date']})" if task.get("due_date") else ""
            
            result += f"\n{status} #{task['id']} {priority_emoji} {task['task']}{due_info}"
        
        return result
    
    def set_preference(self, key: str, value: str) -> str:
        """Set a user preference"""
        self.data["preferences"][key] = value
        self._save_data()
        return f"💡 Saved preference: {key} = {value}"
    
    def get_preference(self, key: str) -> Optional[str]:
        """Get a user preference"""
        return self.data["preferences"].get(key)
    
    def list_preferences(self) -> str:
        """List all preferences"""
        if not self.data["preferences"]:
            return "🔧 No preferences set yet"
        
        result = "🔧 **Your Preferences:**\n"
        for key, value in self.data["preferences"].items():
            result += f"• {key}: {value}\n"
        
        return result
    
    def add_reminder(self, reminder: str, time: str) -> str:
        """Add a timed reminder"""
        reminder_id = len(self.data["reminders"]) + 1
        new_reminder = {
            "id": reminder_id,
            "reminder": reminder,
            "time": time,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        self.data["reminders"].append(new_reminder)
        self._save_data()
        return f"⏰ Added reminder #{reminder_id}: {reminder} at {time}"
    
    def check_reminders(self) -> str:
        """Check for active reminders"""
        now = datetime.now()
        active_reminders = [r for r in self.data["reminders"] if r["active"]]
        
        if not active_reminders:
            return "🔕 No active reminders"
        
        result = "⏰ **Active Reminders:**\n"
        for reminder in active_reminders:
            result += f"• #{reminder['id']}: {reminder['reminder']} at {reminder['time']}\n"
        
        return result
    
    def plan_day(self, date: Optional[str] = None) -> str:
        """Help plan the day"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Get today's tasks
        today_tasks = [t for t in self.data["tasks"] 
                      if not t["completed"] and 
                      (t.get("due_date") == date or t["priority"] == "high")]
        
        # Get preferences for planning
        work_hours = self.get_preference("work_hours") or "9:00-17:00"
        preferred_break = self.get_preference("preferred_break") or "12:00-13:00"
        
        result = f"📅 **Daily Plan for {date}**\n\n"
        result += f"⏰ Work Hours: {work_hours}\n"
        result += f"☕ Break Time: {preferred_break}\n\n"
        
        if today_tasks:
            result += "🎯 **Priority Tasks:**\n"
            for task in today_tasks[:5]:  # Top 5 tasks
                priority_emoji = {"high": "🔥", "medium": "📌", "low": "💡"}.get(task["priority"], "📌")
                result += f"• {priority_emoji} {task['task']}\n"
        else:
            result += "✨ No urgent tasks for today!\n"
        
        # Add reminders for the day
        day_reminders = [r for r in self.data["reminders"] if r["active"]]
        if day_reminders:
            result += f"\n⏰ **Reminders:**\n"
            for reminder in day_reminders:
                result += f"• {reminder['time']}: {reminder['reminder']}\n"
        
        return result
    
    def get_stats(self) -> str:
        """Get productivity statistics"""
        total_tasks = len(self.data["tasks"])
        completed_tasks = len([t for t in self.data["tasks"] if t["completed"]])
        pending_tasks = total_tasks - completed_tasks
        high_priority = len([t for t in self.data["tasks"] if t["priority"] == "high" and not t["completed"]])
        
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        result = "📊 **Your Productivity Stats:**\n"
        result += f"📝 Total Tasks: {total_tasks}\n"
        result += f"✅ Completed: {completed_tasks}\n"
        result += f"⏳ Pending: {pending_tasks}\n"
        result += f"🔥 High Priority Pending: {high_priority}\n"
        result += f"📈 Completion Rate: {completion_rate:.1f}%\n"
        result += f"🔧 Preferences Set: {len(self.data['preferences'])}\n"
        result += f"⏰ Active Reminders: {len([r for r in self.data['reminders'] if r['active']])}\n"
        
        return result

def main():
    assistant = PersonalAssistant()
    
    print("🤖 Personal Assistant Started!")
    print("Type 'help' for available commands or 'quit' to exit")
    
    while True:
        try:
            command = input("\n💬 What can I help you with? ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Your data has been saved.")
                break
            
            elif command.lower() == 'help':
                help_text = """
🤖 **Available Commands:**

**Tasks:**
• add task [task] [priority] [due_date] - Add a new task
• complete [task_id] - Mark task as completed  
• list tasks - Show pending tasks
• list all tasks - Show all tasks including completed

**Preferences:**
• set [key] [value] - Set a preference
• get [key] - Get a preference value
• list preferences - Show all preferences

**Reminders:**
• add reminder [reminder] at [time] - Add a reminder
• check reminders - Show active reminders

**Planning:**
• plan day [date] - Plan your day (default: today)
• stats - Show productivity statistics

**Examples:**
• add task "Buy groceries" high 2024-01-15
• set work_hours 9:00-18:00  
• add reminder "Team meeting" at 14:00
• plan day 2024-01-15
                """
                print(help_text)
            
            elif command.startswith('add task'):
                parts = command[8:].strip().split()
                if len(parts) >= 1:
                    # Extract task name (everything in quotes or first word)
                    if '"' in command:
                        task = re.findall(r'"([^"]*)"', command)[0]
                        remaining = command.split(f'"{task}"')[1].strip().split()
                    else:
                        task = parts[0]
                        remaining = parts[1:]
                    
                    priority = remaining[0] if remaining else "medium"
                    due_date = remaining[1] if len(remaining) > 1 else None
                    
                    print(assistant.add_task(task, priority, due_date))
                else:
                    print("❌ Please specify a task")
            
            elif command.startswith('complete'):
                try:
                    task_id = int(command.split()[1])
                    print(assistant.complete_task(task_id))
                except (IndexError, ValueError):
                    print("❌ Please specify a valid task ID")
            
            elif command == 'list tasks':
                print(assistant.list_tasks())
            
            elif command == 'list all tasks':
                print(assistant.list_tasks(show_completed=True))
            
            elif command.startswith('set'):
                parts = command[3:].strip().split(maxsplit=1)
                if len(parts) == 2:
                    print(assistant.set_preference(parts[0], parts[1]))
                else:
                    print("❌ Usage: set [key] [value]")
            
            elif command.startswith('get'):
                key = command[3:].strip()
                value = assistant.get_preference(key)
                if value:
                    print(f"💡 {key}: {value}")
                else:
                    print(f"❌ Preference '{key}' not found")
            
            elif command == 'list preferences':
                print(assistant.list_preferences())
            
            elif command.startswith('add reminder'):
                # Parse: add reminder "reminder text" at time
                if ' at ' in command:
                    reminder_part, time_part = command.split(' at ')
                    reminder = reminder_part.replace('add reminder ', '').strip().strip('"')
                    time = time_part.strip()
                    print(assistant.add_reminder(reminder, time))
                else:
                    print("❌ Usage: add reminder [text] at [time]")
            
            elif command == 'check reminders':
                print(assistant.check_reminders())
            
            elif command.startswith('plan day'):
                date_part = command[8:].strip()
                date = date_part if date_part else None
                print(assistant.plan_day(date))
            
            elif command == 'stats':
                print(assistant.get_stats())
            
            else:
                print("❌ Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Your data has been saved.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()