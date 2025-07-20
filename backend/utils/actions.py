import webbrowser
import subprocess
import os
import platform
import datetime
import re
import logging
from typing import Dict, Callable, Tuple, Optional
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionHandler:
    def __init__(self):
        """Initialize the action handler with available commands"""
        self.system = platform.system().lower()
        self.actions = {
            # Web actions
            "open_youtube": self.open_youtube,
            "open_google": self.open_google,
            "search_google": self.search_google,
            "open_website": self.open_website,
            
            # System actions  
            "tell_time": self.tell_time,
            "tell_date": self.tell_date,
            "open_calculator": self.open_calculator,
            "open_notepad": self.open_notepad,
            "open_file_manager": self.open_file_manager,
            
            # Media actions
            "play_music": self.play_music,
            "pause_music": self.pause_music,
            "volume_up": self.volume_up,
            "volume_down": self.volume_down,
            
            # System control
            "shutdown": self.shutdown_system,
            "restart": self.restart_system,
            "lock_screen": self.lock_screen,
        }
        
        # Intent patterns for natural language processing
        self.intent_patterns = {
            r"(open|go to|visit).*(youtube|yt)": "open_youtube",
            r"(open|go to|visit).*(google)": "open_google",
            r"(search|google|look up)\s+(.+)": "search_google",
            r"(open|visit|go to)\s+([\w\-\.]+\.[\w]{2,})": "open_website",
            r"(what.?s?.*(time|clock)|tell.*time|current time)": "tell_time",
            r"(what.?s?.*(date|today)|tell.*date|current date)": "tell_date",
            r"(open|start).*(calculator|calc)": "open_calculator",
            r"(open|start).*(notepad|text editor|editor)": "open_notepad",
            r"(open|show).*(file manager|explorer|files|folder)": "open_file_manager",
            r"(play|start).*(music|song|audio)": "play_music",
            r"(pause|stop).*(music|song|audio)": "pause_music",
            r"(volume up|louder|increase volume)": "volume_up",
            r"(volume down|quieter|decrease volume)": "volume_down",
            r"(shutdown|turn off|power off)": "shutdown_system",
            r"(restart|reboot|reset)": "restart_system",
            r"(lock screen|lock computer|lock)": "lock_screen",
        }
    
    def parse_intent(self, user_input: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse user input to determine intended action"""
        user_input = user_input.lower().strip()
        
        for pattern, action_name in self.intent_patterns.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                # Extract parameter if available (like search query or website)
                param = None
                if action_name == "search_google" and len(match.groups()) > 1:
                    param = match.group(2).strip()
                elif action_name == "open_website" and len(match.groups()) > 1:
                    param = match.group(2).strip()
                
                return action_name, param
        
        return None, None
    
    async def execute_action(self, user_input: str) -> Tuple[bool, str]:
        """Execute an action based on user input"""
        action_name, param = self.parse_intent(user_input)
        
        if not action_name:
            return False, "I didn't understand that command."
        
        if action_name not in self.actions:
            return False, f"Action '{action_name}' is not available."
        
        try:
            action_func = self.actions[action_name]
            if param:
                result = await action_func(param)
            else:
                result = await action_func()
            
            return True, result
            
        except Exception as e:
            logger.error(f"Error executing action {action_name}: {e}")
            return False, f"Failed to execute {action_name}: {str(e)}"
    
    # Web Actions
    async def open_youtube(self, param: str = None) -> str:
        """Open YouTube in default browser"""
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube for you!"
    
    async def open_google(self, param: str = None) -> str:
        """Open Google in default browser"""
        webbrowser.open("https://www.google.com")
        return "Opening Google for you!"
    
    async def search_google(self, query: str) -> str:
        """Search Google with the given query"""
        if not query:
            return "What would you like me to search for?"
        
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(search_url)
        return f"Searching Google for: {query}"
    
    async def open_website(self, website: str) -> str:
        """Open a specific website"""
        if not website.startswith(('http://', 'https://')):
            website = f"https://{website}"
        
        webbrowser.open(website)
        return f"Opening {website}"
    
    # Time and Date Actions
    async def tell_time(self, param: str = None) -> str:
        """Tell the current time"""
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}"
    
    async def tell_date(self, param: str = None) -> str:
        """Tell the current date"""
        current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}"
    
    # System Actions
    async def open_calculator(self, param: str = None) -> str:
        """Open calculator application"""
        try:
            if self.system == "windows":
                subprocess.Popen("calc.exe")
            elif self.system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "Calculator"])
            else:  # Linux
                subprocess.Popen(["gnome-calculator"])
            
            return "Opening calculator"
        except:
            return "Could not open calculator"
    
    async def open_notepad(self, param: str = None) -> str:
        """Open text editor"""
        try:
            if self.system == "windows":
                subprocess.Popen("notepad.exe")
            elif self.system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "TextEdit"])
            else:  # Linux
                subprocess.Popen(["gedit"])
            
            return "Opening text editor"
        except:
            return "Could not open text editor"
    
    async def open_file_manager(self, param: str = None) -> str:
        """Open file manager"""
        try:
            if self.system == "windows":
                subprocess.Popen("explorer.exe")
            elif self.system == "darwin":  # macOS
                subprocess.Popen(["open", "."])
            else:  # Linux
                subprocess.Popen(["nautilus"])
            
            return "Opening file manager"
        except:
            return "Could not open file manager"
    
    # Media Actions (Basic)
    async def play_music(self, param: str = None) -> str:
        """Play music (opens default music app)"""
        try:
            if self.system == "windows":
                # Open Windows Media Player or Spotify
                webbrowser.open("https://open.spotify.com")
            elif self.system == "darwin":  # macOS
                subprocess.Popen(["open", "-a", "Music"])
            else:  # Linux
                subprocess.Popen(["rhythmbox"])
            
            return "Starting music player"
        except:
            return "Could not start music player"
    
    async def pause_music(self, param: str = None) -> str:
        """Pause music (basic implementation)"""
        # This is a basic implementation - real pause would need media controls
        return "Music controls would need to be implemented for your specific media player"
    
    async def volume_up(self, param: str = None) -> str:
        """Increase system volume"""
        try:
            if self.system == "windows":
                # Use nircmd or direct Windows API
                return "Volume up command sent (requires additional setup)"
            elif self.system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])
                return "Volume increased"
            else:  # Linux
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%+"])
                return "Volume increased"
        except:
            return "Could not adjust volume"
    
    async def volume_down(self, param: str = None) -> str:
        """Decrease system volume"""
        try:
            if self.system == "windows":
                return "Volume down command sent (requires additional setup)"
            elif self.system == "darwin":  # macOS
                subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])
                return "Volume decreased"
            else:  # Linux
                subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%-"])
                return "Volume decreased"
        except:
            return "Could not adjust volume"
    
    # System Control (Use with caution!)
    async def shutdown_system(self, param: str = None) -> str:
        """Shutdown the system (with confirmation)"""
        return "System shutdown requires manual confirmation for safety"
    
    async def restart_system(self, param: str = None) -> str:
        """Restart the system (with confirmation)"""
        return "System restart requires manual confirmation for safety"
    
    async def lock_screen(self, param: str = None) -> str:
        """Lock the screen"""
        try:
            if self.system == "windows":
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"])
            elif self.system == "darwin":  # macOS
                subprocess.run(["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"])
            else:  # Linux
                subprocess.run(["xdg-screensaver", "lock"])
            
            return "Locking screen"
        except:
            return "Could not lock screen"

# Test the action handler
async def test_actions():
    """Test the action handler"""
    handler = ActionHandler()
    
    test_commands = [
        "open YouTube",
        "search for Python tutorials",
        "what time is it",
        "open calculator",
        "visit github.com"
    ]
    
    for command in test_commands:
        print(f"\nTesting: {command}")
        success, response = await handler.execute_action(command)
        print(f"Result: {response}")

if __name__ == "__main__":
    asyncio.run(test_actions())