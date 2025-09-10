import os
import subprocess
import webbrowser
import logging
from datetime import datetime
from typing import Dict, Callable, Tuple, Any
import platform
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionHandler:
    """Handle system actions and commands"""
    
    def __init__(self):
        """Initialize action handler with available actions"""
        self.system = platform.system().lower()
        self.actions = self._initialize_actions()
        logger.info(f"ActionHandler initialized for {self.system} with {len(self.actions)} actions")
    
    def _initialize_actions(self) -> Dict[str, Callable]:
        """Initialize available actions based on the system"""
        base_actions = {
            # Web actions
            "open_youtube": self._open_youtube,
            "open_google": self._open_google,
            "search_google": self._search_google,
            "open_website": self._open_website,
            
            # Time and date
            "tell_time": self._tell_time,
            "tell_date": self._tell_date,
            "tell_datetime": self._tell_datetime,
            
            # System info
            "system_info": self._system_info,
            "weather": self._weather_placeholder,
            
            # Basic calculations
            "calculate": self._calculate,
        }
        
        # Add system-specific actions
        if self.system == "windows":
            base_actions.update({
                "open_calculator": self._open_calculator_windows,
                "open_notepad": self._open_notepad_windows,
                "open_file_manager": self._open_file_manager_windows,
                "lock_screen": self._lock_screen_windows,
                "volume_up": self._volume_up_windows,
                "volume_down": self._volume_down_windows,
                "volume_mute": self._volume_mute_windows,
            })
        elif self.system == "darwin":  # macOS
            base_actions.update({
                "open_calculator": self._open_calculator_mac,
                "open_finder": self._open_finder_mac,
                "lock_screen": self._lock_screen_mac,
                "volume_up": self._volume_up_mac,
                "volume_down": self._volume_down_mac,
                "volume_mute": self._volume_mute_mac,
            })
        elif self.system == "linux":
            base_actions.update({
                "open_calculator": self._open_calculator_linux,
                "open_file_manager": self._open_file_manager_linux,
                "lock_screen": self._lock_screen_linux,
            })
        
        return base_actions
    
    async def execute_action(self, user_input: str) -> Tuple[bool, str]:
        """
        Execute action based on user input
        
        Returns:
            Tuple of (action_executed: bool, result: str)
        """
        user_input = user_input.lower().strip()
        
        try:
            # Check for direct action matches
            for action_name, action_func in self.actions.items():
                if await self._matches_action(user_input, action_name):
                    logger.info(f"Executing action: {action_name}")
                    result = await self._safe_execute(action_func, user_input)
                    return True, result
            
            # Check for web searches
            if any(phrase in user_input for phrase in ["search for", "look up", "find"]):
                search_query = self._extract_search_query(user_input)
                if search_query:
                    result = await self._safe_execute(self._search_google, search_query)
                    return True, result
            
            # Check for website opening
            if any(phrase in user_input for phrase in ["open", "go to", "visit"]) and any(domain in user_input for domain in [".com", ".org", ".net", "website"]):
                website = self._extract_website(user_input)
                if website:
                    result = await self._safe_execute(self._open_website, website)
                    return True, result
            
            # Check for calculations
            if any(phrase in user_input for phrase in ["calculate", "what is", "what's"]) and any(op in user_input for op in ["+", "-", "*", "/", "plus", "minus", "times", "divided"]):
                calculation = self._extract_calculation(user_input)
                if calculation:
                    result = await self._safe_execute(self._calculate, calculation)
                    return True, result
            
            return False, "I didn't understand that command."
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return False, f"Sorry, I encountered an error: {str(e)}"
    
    async def _matches_action(self, user_input: str, action_name: str) -> bool:
        """Check if user input matches an action"""
        action_keywords = {
            "open_youtube": ["youtube", "open youtube", "go to youtube"],
            "open_google": ["google", "open google", "go to google"],
            "tell_time": ["time", "what time", "current time", "time is it"],
            "tell_date": ["date", "what date", "today's date", "what day"],
            "tell_datetime": ["datetime", "date and time"],
            "open_calculator": ["calculator", "calc", "open calculator"],
            "open_notepad": ["notepad", "text editor", "open notepad"],
            "open_file_manager": ["file manager", "explorer", "files", "open files"],
            "open_finder": ["finder", "open finder"],
            "lock_screen": ["lock", "lock screen", "lock computer"],
            "volume_up": ["volume up", "increase volume", "louder"],
            "volume_down": ["volume down", "decrease volume", "quieter"],
            "volume_mute": ["mute", "silence", "turn off sound"],
            "system_info": ["system info", "computer info", "system information"],
            "weather": ["weather", "temperature", "forecast"],
        }
        
        keywords = action_keywords.get(action_name, [action_name.replace("_", " ")])
        return any(keyword in user_input for keyword in keywords)
    
    async def _safe_execute(self, action_func: Callable, *args) -> str:
        """Safely execute an action function"""
        try:
            if asyncio.iscoroutinefunction(action_func):
                return await action_func(*args)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, action_func, *args)
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            return f"Sorry, I couldn't complete that action: {str(e)}"
    
    def _extract_search_query(self, user_input: str) -> str:
        """Extract search query from user input"""
        patterns = ["search for ", "look up ", "find ", "search "]
        for pattern in patterns:
            if pattern in user_input:
                return user_input.split(pattern, 1)[1].strip()
        return ""
    
    def _extract_website(self, user_input: str) -> str:
        """Extract website from user input"""
        words = user_input.split()
        for word in words:
            if any(domain in word for domain in [".com", ".org", ".net", ".co"]):
                return word
            if word in ["youtube", "google", "github", "stackoverflow", "reddit"]:
                return f"{word}.com"
        return ""
    
    def _extract_calculation(self, user_input: str) -> str:
        """Extract calculation from user input"""
        # Simple extraction - could be improved
        import re
        
        # Look for mathematical expressions
        math_pattern = r'[\d\s+\-*/().]+[+\-*/][\d\s+\-*/().]*'
        matches = re.findall(math_pattern, user_input)
        
        if matches:
            return matches[0]
        
        # Handle word-based math
        user_input = user_input.replace("plus", "+").replace("minus", "-")
        user_input = user_input.replace("times", "*").replace("divided by", "/")
        
        matches = re.findall(math_pattern, user_input)
        return matches[0] if matches else ""
    
    # ============ ACTION IMPLEMENTATIONS ============
    
    # Web Actions
    def _open_youtube(self, *args) -> str:
        """Open YouTube"""
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube in your default browser."
    
    def _open_google(self, *args) -> str:
        """Open Google"""
        webbrowser.open("https://www.google.com")
        return "Opening Google in your default browser."
    
    def _search_google(self, query: str) -> str:
        """Search Google"""
        if not query:
            return "Please provide a search query."
        
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(search_url)
        return f"Searching Google for: {query}"
    
    def _open_website(self, website: str) -> str:
        """Open a website"""
        if not website.startswith(("http://", "https://")):
            website = f"https://{website}"
        
        webbrowser.open(website)
        return f"Opening {website} in your default browser."
    
    # Time and Date Actions
    def _tell_time(self, *args) -> str:
        """Tell current time"""
        current_time = datetime.now().strftime("%I:%M %p")
        return f"The current time is {current_time}."
    
    def _tell_date(self, *args) -> str:
        """Tell current date"""
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        return f"Today is {current_date}."
    
    def _tell_datetime(self, *args) -> str:
        """Tell current date and time"""
        current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
        return f"It is {current_datetime}."
    
    # System Info
    def _system_info(self, *args) -> str:
        """Get system information"""
        import psutil
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return (f"System: {platform.system()} {platform.release()}, "
                   f"CPU: {cpu_percent}%, "
                   f"Memory: {memory.percent}% used, "
                   f"Disk: {disk.percent}% used")
        except ImportError:
            return f"System: {platform.system()} {platform.release()}"
    
    def _weather_placeholder(self, *args) -> str:
        """Weather placeholder (would need API integration)"""
        return "Weather information would require API integration. Check your local weather app or website."
    
    def _calculate(self, expression: str) -> str:
        """Perform calculation"""
        try:
            # Basic safety check
            allowed_chars = set("0123456789+-*/().\s")
            if not all(c in allowed_chars for c in expression):
                return "Invalid calculation expression."
            
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Could not calculate '{expression}': {str(e)}"
    
    # ============ WINDOWS ACTIONS ============
    
    def _open_calculator_windows(self, *args) -> str:
        """Open Windows Calculator"""
        try:
            subprocess.run(["calc"], check=True)
            return "Opening Calculator."
        except Exception as e:
            return f"Could not open Calculator: {str(e)}"
    
    def _open_notepad_windows(self, *args) -> str:
        """Open Windows Notepad"""
        try:
            subprocess.run(["notepad"], check=True)
            return "Opening Notepad."
        except Exception as e:
            return f"Could not open Notepad: {str(e)}"
    
    def _open_file_manager_windows(self, *args) -> str:
        """Open Windows File Explorer"""
        try:
            subprocess.run(["explorer"], check=True)
            return "Opening File Explorer."
        except Exception as e:
            return f"Could not open File Explorer: {str(e)}"
    
    def _lock_screen_windows(self, *args) -> str:
        """Lock Windows screen"""
        try:
            subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
            return "Locking screen."
        except Exception as e:
            return f"Could not lock screen: {str(e)}"
    
    def _volume_up_windows(self, *args) -> str:
        """Increase volume on Windows"""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.utils import AudioUtilities
            from pycaw.api.endpointvolume import IAudioEndpointVolume
            
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            current_volume = volume.GetMasterVolume()
            new_volume = min(0.0, current_volume + 0.1)  # Volume is in negative dB
            volume.SetMasterVolume(new_volume, None)
            
            return "Volume increased."
        except ImportError:
            # Fallback using Windows API
            try:
                subprocess.run(["powershell", "-Command", 
                              "(New-Object -comObject WScript.Shell).SendKeys([char]175)"], 
                              check=True, capture_output=True)
                return "Volume increased."
            except Exception as e:
                return f"Could not increase volume: {str(e)}"
    
    def _volume_down_windows(self, *args) -> str:
        """Decrease volume on Windows"""
        try:
            subprocess.run(["powershell", "-Command", 
                          "(New-Object -comObject WScript.Shell).SendKeys([char]174)"], 
                          check=True, capture_output=True)
            return "Volume decreased."
        except Exception as e:
            return f"Could not decrease volume: {str(e)}"
    
    def _volume_mute_windows(self, *args) -> str:
        """Mute volume on Windows"""
        try:
            subprocess.run(["powershell", "-Command", 
                          "(New-Object -comObject WScript.Shell).SendKeys([char]173)"], 
                          check=True, capture_output=True)
            return "Volume muted/unmuted."
        except Exception as e:
            return f"Could not mute volume: {str(e)}"
    
    # ============ MACOS ACTIONS ============
    
    def _open_calculator_mac(self, *args) -> str:
        """Open macOS Calculator"""
        try:
            subprocess.run(["open", "-a", "Calculator"], check=True)
            return "Opening Calculator."
        except Exception as e:
            return f"Could not open Calculator: {str(e)}"
    
    def _open_finder_mac(self, *args) -> str:
        """Open macOS Finder"""
        try:
            subprocess.run(["open", "-a", "Finder"], check=True)
            return "Opening Finder."
        except Exception as e:
            return f"Could not open Finder: {str(e)}"
    
    def _lock_screen_mac(self, *args) -> str:
        """Lock macOS screen"""
        try:
            subprocess.run(["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"], check=True)
            return "Locking screen."
        except Exception as e:
            return f"Could not lock screen: {str(e)}"
    
    def _volume_up_mac(self, *args) -> str:
        """Increase volume on macOS"""
        try:
            subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"], check=True)
            return "Volume increased."
        except Exception as e:
            return f"Could not increase volume: {str(e)}"
    
    def _volume_down_mac(self, *args) -> str:
        """Decrease volume on macOS"""
        try:
            subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"], check=True)
            return "Volume decreased."
        except Exception as e:
            return f"Could not decrease volume: {str(e)}"
    
    def _volume_mute_mac(self, *args) -> str:
        """Mute volume on macOS"""
        try:
            subprocess.run(["osascript", "-e", "set volume with output muted"], check=True)
            return "Volume muted."
        except Exception as e:
            return f"Could not mute volume: {str(e)}"
    
    # ============ LINUX ACTIONS ============
    
    def _open_calculator_linux(self, *args) -> str:
        """Open Linux Calculator"""
        calculators = ["gnome-calculator", "kcalc", "galculator", "qalculate-gtk"]
        
        for calc in calculators:
            try:
                subprocess.run([calc], check=True)
                return f"Opening {calc}."
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        return "Could not find a calculator application."
    
    def _open_file_manager_linux(self, *args) -> str:
        """Open Linux File Manager"""
        file_managers = ["nautilus", "dolphin", "thunar", "pcmanfm", "nemo"]
        
        for fm in file_managers:
            try:
                subprocess.run([fm], check=True)
                return f"Opening {fm}."
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        return "Could not find a file manager application."
    
    def _lock_screen_linux(self, *args) -> str:
        """Lock Linux screen"""
        lock_commands = [
            ["gnome-screensaver-command", "--lock"],
            ["xdg-screensaver", "lock"],
            ["loginctl", "lock-session"],
        ]
        
        for cmd in lock_commands:
            try:
                subprocess.run(cmd, check=True)
                return "Locking screen."
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        return "Could not lock screen."

# Test function
async def test_action_handler():
    """Test the action handler"""
    handler = ActionHandler()
    
    test_commands = [
        "what time is it?",
        "open youtube",
        "search for python tutorials",
        "what's 2 + 2?",
        "tell me the date",
        "open calculator",
    ]
    
    print("Testing Action Handler...")
    print(f"Available actions: {list(handler.actions.keys())}")
    print("-" * 50)
    
    for command in test_commands:
        executed, result = await handler.execute_action(command)
        print(f"Command: {command}")
        print(f"Executed: {executed}")
        print(f"Result: {result}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_action_handler())