import gradio as gr
from assistant import PersonalAssistant
import re

class WebAssistant:
    def __init__(self):
        self.assistant = PersonalAssistant()
    
    def process_command(self, command: str) -> str:
        """Process commands through the web interface"""
        command = command.strip()
        
        if not command:
            return "Please enter a command"
        
        try:
            if command.startswith('add task'):
                parts = command[8:].strip().split()
                if len(parts) >= 1:
                    if '"' in command:
                        task = re.findall(r'"([^"]*)"', command)[0]
                        remaining = command.split(f'"{task}"')[1].strip().split()
                    else:
                        task = parts[0]
                        remaining = parts[1:]
                    
                    priority = remaining[0] if remaining else "medium"
                    due_date = remaining[1] if len(remaining) > 1 else None
                    
                    return self.assistant.add_task(task, priority, due_date)
                else:
                    return "❌ Please specify a task"
            
            elif command.startswith('complete'):
                try:
                    task_id = int(command.split()[1])
                    return self.assistant.complete_task(task_id)
                except (IndexError, ValueError):
                    return "❌ Please specify a valid task ID"
            
            elif command == 'list tasks':
                return self.assistant.list_tasks()
            
            elif command == 'list all tasks':
                return self.assistant.list_tasks(show_completed=True)
            
            elif command.startswith('set'):
                parts = command[3:].strip().split(maxsplit=1)
                if len(parts) == 2:
                    return self.assistant.set_preference(parts[0], parts[1])
                else:
                    return "❌ Usage: set [key] [value]"
            
            elif command.startswith('get'):
                key = command[3:].strip()
                value = self.assistant.get_preference(key)
                if value:
                    return f"💡 {key}: {value}"
                else:
                    return f"❌ Preference '{key}' not found"
            
            elif command == 'list preferences':
                return self.assistant.list_preferences()
            
            elif command.startswith('add reminder'):
                if ' at ' in command:
                    reminder_part, time_part = command.split(' at ')
                    reminder = reminder_part.replace('add reminder ', '').strip().strip('"')
                    time = time_part.strip()
                    return self.assistant.add_reminder(reminder, time)
                else:
                    return "❌ Usage: add reminder [text] at [time]"
            
            elif command == 'check reminders':
                return self.assistant.check_reminders()
            
            elif command.startswith('plan day'):
                date_part = command[8:].strip()
                date = date_part if date_part else None
                return self.assistant.plan_day(date)
            
            elif command == 'stats':
                return self.assistant.get_stats()
            
            elif command.lower() == 'help':
                return """
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
            
            else:
                return "❌ Unknown command. Type 'help' for available commands."
        
        except Exception as e:
            return f"❌ Error: {e}"
    
    def quick_add_task(self, task: str, priority: str, due_date: str) -> str:
        """Quick task addition through form"""
        if not task.strip():
            return "❌ Please enter a task"
        
        due = due_date if due_date.strip() else None
        return self.assistant.add_task(task, priority.lower(), due)
    
    def get_dashboard(self) -> str:
        """Get dashboard view"""
        result = self.assistant.get_stats() + "\n\n"
        result += self.assistant.list_tasks() + "\n\n"
        result += self.assistant.check_reminders()
        return result

def create_interface():
    web_assistant = WebAssistant()
    
    with gr.Blocks(title="Personal Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 Personal Assistant")
        gr.Markdown("Your AI-powered task manager and daily planner")
        
        with gr.Tabs():
            # Dashboard Tab
            with gr.TabItem("📊 Dashboard"):
                dashboard_output = gr.Textbox(
                    label="Dashboard",
                    lines=20,
                    value=web_assistant.get_dashboard(),
                    interactive=False
                )
                refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="primary")
                refresh_btn.click(
                    fn=web_assistant.get_dashboard,
                    outputs=dashboard_output
                )
            
            # Command Interface Tab
            with gr.TabItem("💬 Commands"):
                gr.Markdown("Enter commands to interact with your assistant")
                with gr.Row():
                    command_input = gr.Textbox(
                        label="Command",
                        placeholder="Type a command (e.g., 'add task \"Buy milk\" high') or 'help'",
                        lines=1
                    )
                    submit_btn = gr.Button("Execute", variant="primary")
                
                command_output = gr.Textbox(
                    label="Result",
                    lines=10,
                    interactive=False
                )
                
                submit_btn.click(
                    fn=web_assistant.process_command,
                    inputs=command_input,
                    outputs=command_output
                )
                
                command_input.submit(
                    fn=web_assistant.process_command,
                    inputs=command_input,
                    outputs=command_output
                )
            
            # Quick Add Tab
            with gr.TabItem("➕ Quick Add Task"):
                with gr.Column():
                    task_input = gr.Textbox(
                        label="Task Description",
                        placeholder="Enter task description"
                    )
                    priority_input = gr.Dropdown(
                        label="Priority",
                        choices=["low", "medium", "high"],
                        value="medium"
                    )
                    due_date_input = gr.Textbox(
                        label="Due Date (optional)",
                        placeholder="YYYY-MM-DD format"
                    )
                    add_btn = gr.Button("Add Task", variant="primary")
                    
                    add_output = gr.Textbox(
                        label="Result",
                        lines=3,
                        interactive=False
                    )
                    
                    add_btn.click(
                        fn=web_assistant.quick_add_task,
                        inputs=[task_input, priority_input, due_date_input],
                        outputs=add_output
                    )
        
        gr.Markdown("""
        ### Quick Start:
        1. Use the **Dashboard** to see your overview
        2. Use **Commands** for full control (type 'help' for all commands)  
        3. Use **Quick Add Task** for fast task creation
        
        ### Example Commands:
        - `add task "Call dentist" high 2024-01-15`
        - `set work_hours 9:00-17:00`
        - `add reminder "Lunch meeting" at 12:30`
        - `plan day`
        - `list tasks`
        """)
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)