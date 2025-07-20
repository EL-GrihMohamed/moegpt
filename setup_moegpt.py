#!/usr/bin/env python3
"""
MoeGPT Project Setup Script
Run this script to create the complete project structure
"""

import os

def create_project_structure():
    """Create the complete MoeGPT project structure"""
    
    # Project structure
    structure = {
        "moegpt": {
            "backend": {
                "models": {
                    "__init__.py": "",
                    "custom_model.py": "",
                    "model_trainer.py": ""
                },
                "api": {
                    "__init__.py": "",
                    "main.py": "",
                    "routes.py": "",
                    "voice_handler.py": ""
                },
                "data": {
                    "training_data.jsonl": "",
                    "processed_data": {}
                },
                "utils": {
                    "__init__.py": "",
                    "data_processor.py": "",
                    "actions.py": ""
                },
                "requirements.txt": "",
                ".env": ""
            },
            "frontend": {
                "public": {
                    "index.html": "",
                    "moegpt-logo.png": ""
                },
                "src": {
                    "components": {
                        "VoiceSelector.jsx": "",
                        "ChatInterface.jsx": "",
                        "MicrophoneButton.jsx": "",
                        "ConversationLog.jsx": ""
                    },
                    "hooks": {
                        "useVoiceRecognition.js": "",
                        "useTextToSpeech.js": ""
                    },
                    "services": {
                        "api.js": "",
                        "voiceService.js": ""
                    },
                    "styles": {
                        "globals.css": ""
                    },
                    "App.jsx": "",
                    "main.jsx": ""
                },
                "package.json": "",
                ".env": ""
            },
            "deployment": {
                "docker-compose.yml": "",
                "Dockerfile.backend": "",
                "Dockerfile.frontend": ""
            },
            "docs": {
                "training_guide.md": "",
                "deployment_guide.md": ""
            },
            "README.md": ""
        }
    }
    
    def create_structure(base_path, structure_dict):
        """Recursively create directory structure"""
        for name, content in structure_dict.items():
            current_path = os.path.join(base_path, name)
            
            if isinstance(content, dict):
                # It's a directory
                os.makedirs(current_path, exist_ok=True)
                print(f"Created directory: {current_path}")
                create_structure(current_path, content)
            else:
                # It's a file
                os.makedirs(os.path.dirname(current_path), exist_ok=True)
                with open(current_path, 'w') as f:
                    f.write(content)
                print(f"Created file: {current_path}")
    
    # Create the structure
    create_structure(".", structure)
    
    print("\n✅ MoeGPT project structure created successfully!")
    print("\nProject structure:")
    print("moegpt/")
    print("├── backend/")
    print("│   ├── models/")
    print("│   │   ├── __init__.py")
    print("│   │   ├── custom_model.py")
    print("│   │   └── model_trainer.py")
    print("│   ├── api/")
    print("│   │   ├── __init__.py")
    print("│   │   ├── main.py")
    print("│   │   ├── routes.py")
    print("│   │   └── voice_handler.py")
    print("│   ├── data/")
    print("│   │   ├── training_data.jsonl")
    print("│   │   └── processed_data/")
    print("│   ├── utils/")
    print("│   │   ├── __init__.py")
    print("│   │   ├── data_processor.py")
    print("│   │   └── actions.py")
    print("│   ├── requirements.txt")
    print("│   └── .env")
    print("├── frontend/")
    print("│   ├── public/")
    print("│   │   ├── index.html")
    print("│   │   └── moegpt-logo.png")
    print("│   ├── src/")
    print("│   │   ├── components/")
    print("│   │   │   ├── VoiceSelector.jsx")
    print("│   │   │   ├── ChatInterface.jsx")
    print("│   │   │   ├── MicrophoneButton.jsx")
    print("│   │   │   └── ConversationLog.jsx")
    print("│   │   ├── hooks/")
    print("│   │   │   ├── useVoiceRecognition.js")
    print("│   │   │   └── useTextToSpeech.js")
    print("│   │   ├── services/")
    print("│   │   │   ├── api.js")
    print("│   │   │   └── voiceService.js")
    print("│   │   ├── styles/")
    print("│   │   │   └── globals.css")
    print("│   │   ├── App.jsx")
    print("│   │   └── main.jsx")
    print("│   ├── package.json")
    print("│   └── .env")
    print("├── deployment/")
    print("│   ├── docker-compose.yml")
    print("│   ├── Dockerfile.backend")
    print("│   └── Dockerfile.frontend")
    print("├── docs/")
    print("│   ├── training_guide.md")
    print("│   └── deployment_guide.md")
    print("└── README.md")
    
    print("\nNext steps:")
    print("1. cd moegpt")
    print("2. Set up your backend environment")
    print("3. Set up your frontend environment")
    print("4. Start developing your MoeGPT chatbot!")

if __name__ == "__main__":
    create_project_structure()